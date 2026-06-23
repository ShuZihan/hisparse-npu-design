## NPU HiSparse 技术路径说明

### 1. 问题

当前 GLM-5 在 NPU 上已经会 sparse attention：`npu_lightning_indexer` 每步选 top-k token，例如 2048 个；`torch_npu.npu_sparse_flash_attention(...)` 只对这些 token 算 attention。它省计算，不省 KV 显存：KV cache 仍按 PA_BSND 全量常驻 NPU。

HiSparse 改的是 KV 存放位置：完整 KV 放 Host，NPU 每层只留一个小 hot buffer。比如 128K 上下文里，每步 attention 只读 2048 个 token，就希望 NPU 只放 4096 个 LRU slot + 1 个 newest slot，而不是放完整 128K。

GPU 能直接做，是因为 FlashMLA sparse decode 的 `indices` 已经是 **device KV cache 坐标**。NPU SFA 当前的 `sparse_indices` 不是物理地址，还要经 `layout_kv` 和 `block_table` 解释。因此 NPU HiSparse 的本质是：**在 SFA 前做地址空间翻译：logical token -> hot location -> SFA 坐标**。由于 GLM-5 当前生产调用已经是 `layout_query="TND"` + `layout_kv="PA_BSND"`，最小改造主线应先保留 PA_BSND，通过重映射 `block_table` 让 SFA 读 hot buffer；BSND hot-slot 直读更接近 GPU 的物理槽清单，但需要改调用形态。

| 项 | 当前 NPU GLM-5 | NPU HiSparse 后 |
|---|---|---|
| KV 在哪 | 全量 KV 在 NPU PA_BSND cache | 全量 KV 在 Host，NPU 只有 hot buffer |
| top-k 是什么 | logical token，例如 token 9001 | 仍然是 logical token |
| SFA 读哪 | `topk_indices + block_table` 指向全量 NPU cache | 改写后的坐标指向 NPU hot buffer |
| 新增工作 | 无 residency | hit/miss、搬 miss、把 logical token 改写成 hot location/SFA 坐标 |

```mermaid
flowchart LR
    I[npu_lightning_indexer<br/>logical top-k: token 9001] --> R[Residency 管理层<br/>查 token 9001 是否 resident]
    H[(Host KV pool<br/>完整 128K KV)] --> C[swap-in<br/>miss 时搬 KV]
    R -->|hit: hot block 3 或 slot 407| A[SFA 坐标转换<br/>hot location -> SFA 坐标]
    R -->|miss: 分配 slot 812| C
    C --> B[(NPU hot buffer<br/>4096 LRU + 1 newest)]
    C --> A
    B --> S[npu_sparse_flash_attention]
    A --> S
```

只记两个对象：indexer 输出 **logical token**，例如 `9001`；SFA 最终要读 **hot buffer 里的 KV**。方案 A 读的是 hot block 加 offset，方案 B 读的是 hot slot。

### 2. token 9001 怎么走

假设 batch=2、decode、topk=2048，request 0 的第 k 个 top-k 是 logical token `9001`。

| 路径 | token 9001 后续怎么变成 KV 地址 | 关键差异 |
|---|---|---|
| GPU FlashMLA + HiSparse | `swap_in_selected_pages(...)` 查 hot buffer；miss 时从 Host 搬到 GPU device buffer；返回 `top_k_device_locs[0,k]=x`；SGLang 传 `indices=page_table_1.unsqueeze(1)`；FlashMLA kernel 对 `x` 做除法/取模，读 `k_cache` | `x` 已经是 device KV cache 坐标，FlashMLA 不需要理解 Host、LRU、miss |
| 当前 NPU SFA | `npu_lightning_indexer` 输出 `topk_indices[0,k]=9001`；SFA 先算 `s2Idx=sparse_indices+s2StartIdx`；当前 `layout_kv="PA_BSND"`，再用 `s2Idx/blockSize` 查 `block_table`；最后读全量 NPU KV cache | `9001` 是 S2 token 位置，不是物理地址，也不是 hot slot |
| NPU HiSparse 后 | residency 管理层先查 token `9001` 是否在 hot buffer；方案 A 找到它所在的 hot block，方案 B 找到它所在的 hot slot | 多了 `logical token -> hot location -> SFA 坐标` 两段 |

最大差异：

| | GPU FlashMLA | NPU SFA |
|---|---|---|
| attention 输入 index 的语义 | 已经是 device KV 坐标 | 仍是 SFA S2 空间坐标 |
| 是否需要改 attention 调用契约 | 很少，直接给 `indices` | 必须把 hot location 编成 SFA 能读的坐标 |
| 是否依赖 page table | sparse FlashMLA 不依赖 | PA_BSND 依赖；BSND 不依赖 |

### 3. NPU 适配拆成两段

| 转换 | 输入 | 输出 | 消费方 |
|---|---|---|---|
| Residency | logical top-k，例如 `[9001, 17, ...]` | 方案 A 输出 hot block/page 映射；方案 B 输出 hot slot，例如 `[407, 12, ...]` | SFA 坐标转换层 |
| SFA 坐标转换层 | A 的 hot block/page 映射，或 B 的 hot slot | A 生成 remapped `block_table`；B 生成 BSND 的 S 轴 index | `npu_sparse_flash_attention` |

Residency 回答“token 在不在 NPU 上、miss 放到 hot buffer 哪里”；SFA 坐标转换层回答“怎样把这个 hot buffer 位置翻译成 SFA 能读的坐标”。方案 A 和 B 的差异就在第二步：A 不把 token slot 直接交给 SFA，而是改 `block_table`；B 直接把 hot slot 当 S 轴下标交给 SFA。

### 4. 方案 A 和 B 到底差在哪

先固定一个例子：request 0 的 top-k 里有 logical token `9001`，PA block size 是 128。这个 token 在原始长序列里的位置可以拆成：

| 值 | 计算 | 含义 |
|---|---|---|
| logical token | `9001` | indexer 选中的原始 token |
| logical block | `9001 // 128 = 70` | token 9001 属于第 70 个 PA block |
| offset in block | `9001 % 128 = 41` | token 9001 是这个 block 内第 41 个 token |

方案 A 和 B 的区别不是“都拿 slot 407，只是名字不同”。真正区别是：**A 让 SFA 按 logical token 查表读 hot block；B 让 SFA 直接按 hot slot 读小 KV cache。**

#### A：PA_BSND `block_table` 重映射

A 保留当前 GLM-5 的调用形态：`layout_query="TND"`、`layout_kv="PA_BSND"`、继续传 `block_table`。SFA 看到的 `sparse_indices` 仍然可以是 `9001`，也就是原始 S2 位置。

如果 Residency 管理层决定把 logical block 70 搬到 hot buffer 的 hot block 3，那么 SFA 坐标转换层写：

```python
block_table[request_0, 70] = 3
sparse_indices[request_0, k] = 9001
```

SFA 内部读 KV 时做的是：

```python
s2Idx = sparse_indices + s2StartIdx   # 假设 decode 下 s2StartIdx = 0，则 s2Idx = 9001
blkTableIdx = 9001 // 128             # 70
blkTableOffset = 9001 % 128           # 41
realkeyOffset = block_table[0, 70] * 128 + 41
              = 3 * 128 + 41
              = 425
```

所以 A 读到的是 hot PA cache 里的第 `425` 个物理 token。注意这里没有“slot 407”这个概念；A 的 resident 单位是 **block/page**。如果 token 9001 miss，通常要把它所在的 128-token block 搬进来。

#### B：BSND hot-slot 直读

B 不走 `block_table`。它把 NPU hot buffer 做成一个小的 BSND KV cache，例如 `key/value=[B,4097,1,D]`。如果 token 9001 被放在 request 0 的 hot slot 407，SFA 坐标转换层写：

```python
sparse_indices[request_0, k] = 407
block_table = None
layout_query = "BSND"
layout_kv = "BSND"
```

SFA 内部读 KV 时做的是：

```python
s2Idx = sparse_indices + s2StartIdx   # 假设 decode 下 s2StartIdx = 0，则 s2Idx = 407
realkeyOffset = request_id * s2Size + s2Idx
              = 0 * 4097 + 407
              = 407
```

所以 B 读到的是 hot buffer 的第 `407` 个 slot。B 的 resident 单位是 **token slot**。它没有 page 放大，但必须把当前 GLM-5 的 query/rope/output 调用形态从 TND+PA_BSND 切到 BSND。

#### 一张表记住

| 方案 | SFA 看到的 KV | 坐标怎么填 | 优点 | 风险/约束 | 结论 |
|---|---|---|---|---|---|
| A：PA_BSND `block_table` 重映射 | 当前 GLM-5 形态：`layout_query="TND"`、`layout_kv="PA_BSND"`、`block_table` | `sparse_indices=9001`，`block_table[0,70]=3`，SFA 读 `3*128+41=425` | query/rope/output 形态改动最少 | resident 单位是 block/page；block_size=128 时，一个 miss token 可能要求搬整个 128-token block | 当前主线，先做 golden 并量化 page 放大 |
| B：BSND hot-slot 直读 | `key/value=[B,4097,1,512]`，`key_rope=[B,4097,1,64]`，`block_table=None`，`layout_kv="BSND"`，`layout_query="BSND"` | token `9001` 在 request 0 slot 407，则 `sparse_indices=407`；BSND 分支用 `boIdx*s2Size+s2Idx` 直接读 slot | 最像 GPU；token-wise；无 page 放大 | 当前生产不是 BSND；query/rope/output reshape 都要切；SFA 会做 `sparse_indices+s2StartIdx`；pytest/eager golden 只覆盖 BSND K=16/32，GLM-5 K=2048 + MLA rope + graph 参数组合需 golden | 优化/备选方案，不作为最小改造主线 |
| C：改 SFA | 新 hot-slot 调用契约 | `sparse_indices` 直接表示 hot slot，或支持 FlashMLA-style encoded KV index | 语义最干净 | 要改 `ops-transformer/attention/sparse_flash_attention` 的 op host、tiling、kernel、测试和分发 | A/B 都对不齐 baseline 时最后备选 |

PA_BSND 重映射有两个变体：

| 变体 | 做法 | 代价 |
|---|---|---|
| 保留原始 logical S2 | `sparse_indices` 继续填 9001，`block_table` 覆盖原始 128K 序列的 block 空间 | table 语义简单，但 hot buffer 承受 page 放大 |
| compact hot S2 | top-k token 重写到小 hot S2 空间，`block_table` 只覆盖 hot blocks | 每步维护 logical token 到 compact token/page 的映射 |

### 5. miss KV 怎么搬

SFA 方案解决“读哪里”；搬运方案解决“miss 的 KV 怎么从 Host 到 NPU hot buffer”。

| 搬运方案 | 流程 | 优点 | 风险/代价 |
|---|---|---|---|
| kernel 内搬运 | 一个 AscendC residency kernel 读 `topk_indices`，查 resident table；hit 直接输出 hot block 或 hot slot；miss 选 evict location，例如 hot block 3 或 slot 812；kernel 内从 Host KV pool 搬 KV；输出 hot location 给 SFA 坐标转换层 | 一次 launch 内完成 hit/miss、LRU、miss copy、hot location 输出 | 必须实测 AscendC kernel 能否高效读 Host memory：Host 注册方式、kernel 内 `DataCopy`、带宽、CANN Graph |
| Host staged DMA | judge kernel 只输出 miss list、evict location、hit location；Host 读 miss list；Host 发 H2D copy；finalize 补齐最终 `sparse_indices` / `block_table`；再调 SFA | 更稳，不依赖 kernel 直读 Host | 每步多一次 device-host-device 控制链，Graph 更容易被切开 |

GPU 已经走 kernel 自搬：CUDA kernel 在 miss 时写 `req_top_k_device_locs`，并从 `host_cache_k` 拷到 `device_buffer_k`。

现有 `transfer_kv_dim_exchange` 只能作为 staged DMA 参考，不是完整答案。它是 Host API，要求 5D tensor，按 page 做 CPU 循环并发 `aclrtMemcpy2dAsync`；适合 PA/page hot buffer。如果走 BSND token slot，需要新的 token-wise staged copy。

### 6. 端到端例子

设 batch=2，每个 request 的 hot buffer 容量约等价于 4096 个 LRU token + 1 个 newest token，每步 topk=2048。方案 A 可以把这 4096 个 token 容量组织成 32 个 128-token hot block；方案 B 则组织成 4096 个 token slot。request 0 有 1700 hit / 348 miss，request 1 有 1800 hit / 248 miss，总 miss=596。

| 路径 | 端到端过程 |
|---|---|
| PA_BSND remap + Host staged DMA | indexer 输出 `[2,2048]` logical tokens；judge kernel 输出 miss list 和 evict hot block；Host staged DMA 搬 miss 对应 hot pages；比如 token 9001 属于 logical block 70，SFA 坐标转换层写 `block_table[0,70]=3`，SFA 用 `sparse_indices=9001` 读 hot PA cache 的 `3*128+41` |
| BSND hot-slot + kernel 内搬运 | `npu_lightning_indexer` 输出 `[2,2048]` logical tokens；residency kernel 查两个 request 的 4096 个 hot slots；3500 个 hit 直接写 hot slot，例如 `9001 -> 407`；596 个 miss 分配 evict slot 并从 Host 搬 KV；SFA 坐标转换层写 `sparse_indices=407`；SFA 用 `[B,4097,1,D]` hot buffer 算 attention |

PA_BSND 方案更贴近现有 GLM-5 调用形态，但若 `block_size=128`，596 个 miss token 可能对应很多 128-token page，实际收益要靠 page 放大统计决定。BSND 方案里，SFA 不知道 Host 和 LRU，只看到一个小 KV cache 和一张 slot 清单，但要先改掉当前 TND+PA_BSND 调用形态。

### 7. 状态模型

每个 `(request, layer, logical token)` 的状态：

```mermaid
stateDiagram-v2
    [*] --> HostOnly: prefill/backup 后完整 KV 在 Host
    HostOnly --> SwappingIn: top-k 选中且 hot buffer miss
    SwappingIn --> Resident: Host->NPU copy 完成
    Resident --> Resident: top-k 命中，更新 LRU
    Resident --> HostOnly: 被 LRU evict
    Resident --> NewestSlot: 当前 decode 新 token写入 reserved slot
    NewestSlot --> HostOnly: backup 到 Host 且进入普通 residency 管理
```

attention 只应该读 `Resident` 或 `NewestSlot`。`HostOnly` token 被 top-k 选中时，必须先 swap-in，不能让 SFA 直接去原始 logical 位置读。

### 8. 开发清单和顺序

| 模块 | 谁调用它 | 输入 | 输出 |
|---|---|---|---|
| Host KV pool | prefill/backup/驻留管理层 | 完整 KV | Host 上按 request/layer/token 可寻址的 KV |
| NPU hot buffer | swap-in/SFA | 固定容量 slot，例如 4096+1 | SFA 可读的小 KV cache |
| Residency table | residency kernel/驻留管理层 | logical token | A：hot block/page；B：hot slot |
| LRU/evict | residency kernel/驻留管理层 | hit/miss 访问 | evict block/page 或 evict slot |
| Swap-in backend | 驻留管理层 | miss token、evict location | KV 被搬到 hot buffer |
| SFA 坐标转换层 | attention backend | hot_locations | A：remapped `block_table`；B：BSND `sparse_indices` |
| Golden tests | 开发和回归 | 全量常驻 baseline 输出、HiSparse 输出 | 数值差异和性能拆分 |

| 顺序 | 做什么 | 要回答的问题 |
|---|---|---|
| P0a | 跑通 `sgl-kernel-npu` 最小自定义算子 | 能不能新增 NPU 驻留管理 kernel |
| P0b | Host 直读探针（第一周并行） | kernel 内 `DataCopy` 能不能读 registered/unified Host，带宽和 Graph 是否可接受 |
| P1 | SFA 寻址 golden：先 PA_BSND remap，再 BSND hot-slot | A/B 不改 SFA 时能否读 hot buffer，page 放大和 BSND 生产参数组合是否可接受 |
| P2 | 写 Python residency reference + judge-only kernel | 先把 hit/miss/LRU 行为跑对，并在 device 上产出 miss list 和 hot_locations |
| P3 | 按 P0b 结论补搬运路径 | P0b 通过则 kernel 自搬；不通过则 Host staged DMA |
| P4 | 接入 swap-in + SFA 坐标转换层 | 单层对齐全量常驻 baseline |
| P5 | 多层 decode、CANN Graph、CP/PD、异常回收验证 | 61 层、batch>1、长上下文稳定对齐 |

### 9. 源码锚点

| 结论 | 源码 |
|---|---|
| GPU HiSparse 把 top-k 交给 `swap_in_selected_pages(...)` | [`nsa_backend.py`](../sglang-v0.5.12/python/sglang/srt/layers/attention/nsa_backend.py#L1612) |
| GPU 最终传 `flash_mla_with_kvcache(indices=page_table_1.unsqueeze(1))` | [`nsa_backend.py`](../sglang-v0.5.12/python/sglang/srt/layers/attention/nsa_backend.py#L1817) |
| FlashMLA `indices` shape 是 `[batch_size, seq_len_q, topk]` | [`flash_mla_interface.py`](../FlashMLA/flash_mla/flash_mla_interface.py#L85) |
| FlashMLA kernel 把 index 拆成 `block_idx` 和 `idx_in_block` | [`kernel.cuh`](../FlashMLA/csrc/sm100/decode/head64/kernel.cuh#L697) |
| GPU HiSparse miss 分支写 `req_top_k_device_locs`，并做 Host->Device copy | [`hisparse.cuh`](../sglang-v0.5.12/python/sglang/jit_kernel/csrc/hisparse.cuh#L347), [`#L371`](../sglang-v0.5.12/python/sglang/jit_kernel/csrc/hisparse.cuh#L371) |
| NPU GLM-5 当前 SFA 调用是 `layout_query="TND"`、`layout_kv="PA_BSND"` | [`ascend_backend.py`](../sglang-v0.5.12/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L949) |
| NPU 当前 `block_table` 来自 `req_to_token[:, ::page_size] // page_size` | [`ascend_backend.py`](../sglang-v0.5.12/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L359) |
| SFA 先做 `sparse_indices + s2StartIdx` | [`sparse_flash_attention_service_vector_mla.h`](../ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h#L203) |
| SFA PA_BSND 用 `blockTableGm[...] * blockSize + offset` | [`sparse_flash_attention_service_vector_mla.h`](../ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h#L225) |
| SFA BSND 直接用 `boIdx * s2Size + s2Idx` | [`sparse_flash_attention_service_vector_mla.h`](../ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h#L231) |
| 非 PA_BSND 要求 `layout_kv == layout_query`，且 `block_table` 为空 | [`sparse_flash_attention_tiling.cpp`](../ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp#L975), [`#L1829`](../ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp#L1829) |
| pytest/eager golden 的 BSND 用例 K=16/32，K=2048 golden 用例走 PA_BSND | [`sparse_flash_attention_paramset.py`](../ops-transformer/attention/sparse_flash_attention/tests/pytest/sparse_flash_attention_paramset.py#L12) |
| PA_BSND block size 要 16 对齐，最大 1024 | [`sparse_flash_attention_tiling.cpp`](../ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp#L1434) |
| `transfer_kv_dim_exchange` 是 5D page 粒度 Host API | [`transfer_kv_dim_exchange.cpp`](../sgl-kernel-npu/csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp#L28) |

### 10. 待确认问题

- BSND 方案下，decode 主路径的 `s2StartIdx` 是否恒为 0；如果不是，hot slot 坐标要怎么扣基址。
- BSND 方案下，`query`、`query_rope`、`attention_out` 从 TND 切到 BSND 后，现有 GLM-5 wrapper 是否能无损接回。
- newest slot 是否会被 top-k 选中；如果会，`actual_seq_lengths_kv` 必须覆盖 slot 4096。
- PA_BSND 方案下，page 放大后 hot buffer 和 H2D 带宽是否仍然划算。
- AscendC kernel 内 Host 直读是否可用，带宽是否足够，能否进 CANN Graph。
- Host staged DMA 方案下，miss list 回 Host、H2D copy、finalize sparse indices 三步的同步成本是多少。

## 第二部分：NPU HiSparse 适配方案

NPU 适配目标：当前 GLM-5 已经用 `npu_lightning_indexer + npu_sparse_flash_attention` 省计算，但 KV 仍按 PA_BSND 全量常驻 NPU。HiSparse 要补的是 **sparse KV residency**：完整 KV 放 Host，NPU 只留 hot buffer，top-k miss 时再把对应 KV 搬回 NPU。

这件事卡在三处：①谁搬 KV，②SFA 怎么读 hot buffer，③host 读和 staged DMA 能否进图并保持收益。后文所有源码、CANN API、伪代码都围绕这三处。

### 2.1 现状与目标

**当前 NPU GLM-5/SFA 路径**已有完整的 sparse attention 推理：`npu_lightning_indexer` 选 top-k → `npu_sparse_flash_attention` 做 token 级 sparse attention。但 KV cache 全量常驻 NPU（PA_BSND layout, block_size=128），只省计算量不省显存。注意：这里的模型类名/历史目录名里可能出现 DSA/NSA，但 GLM-5 在 NPU 上实际进入的是 SFA attention op，不是 `SparseFlashMla` 或压缩 DSA op。

| 维度 | GPU NSA / FlashMLA | NPU GLM-5 / SFA |
|------|----------|---------|
| Indexer | FP8 paged MQA logits + fast_topk | `npu_lightning_indexer` |
| Attention | `flashmla_kv` / `flashmla_sparse` / tilelang / FA3 | `npu_sparse_flash_attention` |
| KV layout | flat token pool, page_size=1 | PA_BSND, block_size=128 |
| HiSparse | 有（swap-in kernel） | **无** |
| KV 常驻 | 启用 HiSparse 时不常驻 | 全量常驻 |

缺的是驻留管理层（residency runtime），也就是维护“哪些 token 已在 hot buffer、哪些 miss、驱逐谁、从 Host 搬到哪里”的运行时逻辑。昇腾官方支持 device 访问 registered host memory（A3/A2 √，§2.2.2），但 HiSparse 需要更具体的能力：**AscendC kernel 内 `DataCopy` 直读 Host**。现有 NPU 路径只看到普通 `pin_memory` 和 host 侧 staged DMA（`transfer_kv_dim_exchange`），没有 kernel 内直读 Host 先例（§2.2.1）。P0b 要验证的是这个用法能否成立、带宽够不够。

**目标**：KV cache 不全量常驻 NPU，只保留 hot buffer，全量 KV 放 host，按需 swap-in。

### 2.2 主线方案

主线拆成两个独立问题：

- **搬运**：路径①让 AscendC residency kernel 同时完成 hit/miss、LRU、miss copy；路径②让 kernel 只输出 miss list，Host 侧 staged DMA 再搬 KV。路径①性能更好，路径②是必须先跑通的基线。
- **寻址**：当前方案 A 保持 GLM-5 生产调用契约（`layout_query="TND"`、`layout_kv="PA_BSND"`、`block_table`），把 `block_table` 重映射到 hot buffer block。方案 B 才把 hot buffer 做成 BSND 并把 hot slot 填进 `sparse_indices`；方案 C 才改 SFA。

```
每步 decode (per layer):
  npu_lightning_indexer → top_k_tokens [B, 2048]
  → AscendC residency kernel:
      输入: top_k_tokens, host_kv_devPtr, hot_buffer, lru_state
      kernel 内部: hit/miss 判定 → LRU 更新 → miss DataCopy Host→hot_buffer
      输出: hot_locations（A=hot block/page 映射；B=hot slot list）
  → SFA 坐标转换层:
      输入: logical top-k, hot_locations
      输出: remapped sparse_indices / block_table
  → npu_sparse_flash_attention(           # 方案 A：保持 PA_BSND，不改算子
      query=q_nope, query_rope=q_pe,       # 保持当前 TND query 形态
      key=hot_pa_cache, value=hot_pa_cache,
      key_rope=hot_pa_rope,
      sparse_indices=<原始 logical S2 或 compact hot S2>,
      block_table=<指向 hot buffer block 的 remapped table>,
      layout_query="TND", layout_kv="PA_BSND", sparse_block_size=1)
```

P0b 只实测三件事：kernel 内 `DataCopy` 能否读 Host devPtr、EP 场景下带宽是否可接受、含 Host read 的 kernel 能否被 CANN Graph capture。分叉如下：

| 分叉 | 条件 | 架构后果 |
|------|------|----------|
| **路径 ①：kernel 内搬运** | kernel 内 DataCopy 直读成立、带宽可接受 | residency kernel 内完成判定和搬运，上面的伪代码成立 |
| **路径 ②：staged DMA** | 直读不成立 或 带宽过高 | kernel 内只算 miss list，搬运退到 kernel 外，用 `aclrtMemcpyAsync` / `transfer_kv_dim_exchange` 做显式 H2D |

路径②不是失败后才做的补救，而是可运行基线：hit/miss 判定不依赖 Host 直读，可先写；P0b 只决定是否把 miss copy 也并进 kernel（§2.7.3.1 / §2.7.5）。

**路径 ② 架构**：

```
每步 decode (per layer):
  npu_lightning_indexer → top_k_tokens [B, 2048]
  → AscendC judge kernel（只判定，不搬运）:
      输入: top_k_tokens, device_buffer_tokens, lru_state
      输出: miss_list [B, M], evict_locations [B, M], hot_locations（hit 部分已填）
  → host 侧发起 staged H2D（搬运 API 取决于 hot buffer layout，见下）:
      aclrtMemcpyAsync(hot_buffer[evict_locations] ← host_pool[miss_list], H2D, copy_stream)
      回填 hot_locations 的 miss 部分
  → SFA 坐标转换层生成 remapped block_table / sparse_indices
  → npu_sparse_flash_attention(...)
```

路径②代价是每步多一趟 device→host→device 控制链，可能打断 CANN Graph 单图捕获。搬运 API 取决于 hot buffer layout：`transfer_kv_dim_exchange` 强制 `device_k.dim()==5`、按 page 搬全 layer，适合方案 A 的 PA/page hot buffer；方案 B 的 BSND/token hot buffer 需要新写 staged copy（逐 token `aclrtMemcpyAsync` 或新算子）。详见 §2.7.5。

#### 2.2.1 代码依据

两处源码给出路径①/②分叉依据：调用侧是本仓库 `sglang-v0.5.12` NPU 后端；实现侧是外部算子包 `../sgl-kernel-npu/`（tag `2026.05.01`），包含真实 AscendC/host 源码。

**(1) NPU 全程没有 host-register，且代码主动绕开它。** 分配器派发表把 NPU 显式路由到普通 pinned 内存，而非 CUDA 的 `cudaHostRegister`（[memory_pool_host.py:147](../sglang-v0.5.12/python/sglang/srt/mem_cache/memory_pool_host.py#L147)）：

```python
ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,   # CUDA 默认走 cudaHostRegister
    {
        "npu": alloc_with_pin_memory,    # NPU 走普通 pin_memory，不注册
        "musa": alloc_with_pin_memory,
    },
)
```

`is_pin_memory_available()` 在没有 CUDA 时直接返回 `False`（`utils/common.py:656`）。调用侧 `grep aclrtHostRegister`、实现侧 `sgl-kernel-npu` 全仓 `grep -i "HostRegister|aclrtMallocHost|ACL_HOST|svm|unified memory"` 均无功能性命中；唯一 "zero-copy" 在 DeepEP MoE 的 device↔device RDMA/IPC ring-buffer。结论：NPU 侧没有 kernel 内直读 host 先例，P0b 验证的是官方能力在 AscendC `DataCopy` 下的具体可用性。

**(2) 既有 NPU host↔device KV 搬运 = 纯 host 侧 staged DMA，且整支算子里没有 kernel。** host 侧 KV 的搬运委托给外部算子 `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`，带 `H2D` / `D2H` 方向参数（`memory_pool_host.py:499/507/606/614`）。它在 `sgl-kernel-npu` 里的实现（[transfer_kv_dim_exchange.cpp](../sgl-kernel-npu/csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp)）说明了路径②的真实形态：它不是 AscendC kernel，而是 `HOST_API`。

```cpp
// transfer_kv_dim_exchange.cpp（节选，H2D 分支）
auto device_indices_cpu = device_indices.cpu();   // ① 索引先拉回 CPU
auto host_indices_cpu   = host_indices.cpu();
const int64_t num_pages = device_indices.size(0) / page_size;
for (const auto i : c10::irange(num_pages)) {       // ② 在 CPU 上逐页 for 循环
    auto device_page_index = device_indices_cpu[i*page_size].item<int64_t>() / page_size;
    auto host_page_index   = host_indices_cpu[i*page_size].item<int64_t>() / page_size;
    void *device_k_ptr = device_k[0][device_page_index].data_ptr();
    void *host_k_ptr   = host_k[host_page_index][0].data_ptr();
    aclrtMemcpy2dAsync(device_k_ptr, k_device_pitch, host_k_ptr, k_host_pitch,  // ③ 每页一发 2D DMA
                       k_width, height, ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
    // device_v 同理再发一发
}
```

这段代码说明三点：索引先 `.cpu()` 回主机；页号配对是 CPU 串行 for；每页一次 `aclrtMemcpy2dAsync`，用 pitch 一次搬该页全部 layer（device `[layer, dev_page, page, head, dim]` ↔ host `[host_page, layer, page, head, dim]`）。它是路径②现成通路，但无法塞进单个 device kernel。

**(3) 可改的 AscendC 源码不在调用仓库，在外部包 `sgl-kernel-npu`。** 本仓库 `sglang-v0.5.12` 全树 `grep DataCopy / __aicore__ / LocalTensor / TPipe` 零命中，所有 NPU 自定义算子来自外部 pip 包 `sgl_kernel_npu`。该包源码已拉取到 `../sgl-kernel-npu/`（tag `2026.05.01`，即 sglang `scripts/ci/npu/npu_ci_install_dependency.sh:71` 钉定的版本），结构如下：

```
sgl-kernel-npu/
├── csrc/                          # 每个算子一目录，op_host(tiling+下发) + op_kernel(AscendC kernel)
│   ├── cache_location_assign/     # ← 最接近 HiSparse 驻留表写入的现成 AscendC 算子(见下 (5))
│   ├── assign_cache_op/           # device↔device 范围拷贝 kernel
│   ├── transfer_kv_dim_exchange/  # host↔device staged DMA(纯 host_api，见 (2))
│   └── lightning_indexer/         # top_k 索引(本仓库自带 AscendC 实现，见 (4))
├── python/sgl_kernel_npu/         # python 封装 + torch.ops 注册
├── build.sh / CMakeLists.txt / cmake/   # ← P0a 要打通的本地构建入口
└── tests/                         # 算子单测，可作 P0a 验证与新算子的模板
```

写 residency kernel 的落点是 `sgl-kernel-npu/csrc/<new_op>/`（`op_host` tiling + `op_kernel` AscendC），再经 `build.sh`、`python/` 注册和 `tests/` 单测进入 `torch.ops.npu.*`。P0a 可直接用 `csrc/helloworld/` 模板。方案 C 改 SFA 则落在 `ops-transformer/attention/sparse_flash_attention`，源码可读但需自带构建分发，并确认与生产 `torch_npu` 二进制对齐。

**(4) SFA 稀疏分页寻址已在用（方案 A 的接入点）。** `npu_sparse_flash_attention` 真实调用参数是 `layout_kv="PA_BSND"` + `block_table` + `sparse_block_size=1` + `sparse_mode=3` + `attention_mode=2`（[ascend_backend.py:949](../sglang-v0.5.12/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L949)），`block_table` 由 `req_to_token[...][:, ::page_size] // page_size` 构造（`:359`）。`lightning_indexer` 有真实 AscendC 实现：`sgl-kernel-npu/csrc/lightning_indexer/op_kernel/` 约 2262 行，README 说明仅 Atlas A3 支持、默认 `sparse_count=2048`、key 支持 `PA_BSND` PageAttention 布局。SFA 源码不在 `sgl-kernel-npu`，但在 `ops-transformer/attention/sparse_flash_attention` 可读；方案 C 仍是最后备选，因为要自带构建分发。

**(5) 现成 AscendC 算子 `cache_location_assign` 给出了 residency kernel 的可复用编码模式。** 这是整个包里语义最接近 HiSparse 驻留逻辑的 AscendC kernel（[cache_loc_assign_kernel.cpp](../sgl-kernel-npu/csrc/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp)）：它做 token-pool↔cache-location 的索引映射（纯 bookkeeping，无 KV 数据搬运）。它把 §2.7 的伪代码落到了真实写法，也暴露了 NPU 的硬约束：

| §2.7 设计点 | `cache_loc_assign` 的真实写法 | 对 residency kernel 的含义 |
|--|--|--|
| 并行单元 | `coreId = AscendC::GetBlockIdx()`,按 row 切给各 vector core(`rowNumNoTail`+`tailNum` tiling) | 是 **per-core 分块**,不是 GPU 的 per-request warp;一个 core 串行处理分到的多行 |
| UB 管理 | 显式 `TPipe` + 7 个 `TBuf<VECCALC>` + `TQue<VECIN,BUFFER_NUM=2>` 手工分配 | 印证 §2.7.4 的 UB 预算必须手算,double-buffer 要显式声明 |
| H2D/UB 搬运 | `DataCopy(ub, gm, count)` + `DataCopyPad(gm, ub, DataCopyExtParams{...})` | MTE 搬的是**规则块/对齐 pad**,不是任意索引 scatter |
| pipe 同步 | 手动 `SetFlag/WaitFlag<HardEvent::MTE2_V>`(读完再算)、`<HardEvent::V_MTE3>`(算完再写) | 印证 §2.8 的 event 模型:流水级间靠 HardEvent 显式卡 |
| **按索引 scatter/gather** | **`for j: cache=ubCacheLoc.GetValue(idx+j); tokenPoolLocal.SetValue(j, cache)`** —— 逐元素标量 `GetValue/SetValue` | **决定性约束**：NPU 没有 warp 并行 scatter；按 miss_list 把零散 token 写到 hot slot，会退化为 Scalar 逐元素循环。这正是 hit 判定必须交给 Vector 批量比较、而非 Scalar 逐个比的根因 |

结论：新 kernel 可复用 `cache_location_assign` 的 tiling/UB/DataCopy/HardEvent 骨架；同时必须规避逐元素 `SetValue` 串行，把判定交给 Vector、搬运交给 MTE 整块 copy。

#### 2.2.2 官方文档依据（CANN 8.5 acl API）

CANN 8.5 acl API 明确支持 device 访问 registered host memory；P0b 验证的是 **AscendC kernel 内 `DataCopy` 是否能使用这类地址**。

| API / 类型 | 官方原文（节选） | 对 P0b 的意义 |
|--|--|--|
| `aclrtHostRegisterType`（枚举） | `ACL_HOST_REGISTER_MAPPED = 0` // "the host memory is mapped and registered as **accessible to the device, including read and write**" | "device 可读写 host 内存"被官方明文背书——这是路径①的能力基础 |
| 产品支持矩阵 | Atlas A3 训练/推理 √、A2 训练/推理 √（200I/500 A2、纯推理卡 ☓） | 正好覆盖目标硬件 A3/A2 |
| `aclrtHostRegister` / `aclrtHostRegisterV2` | "Maps and registers the host memory as a memory address that can be accessed by the device" | host 侧注册入口；V2 配 `ACL_HOST_REG_MAPPED`(0x2) / `ACL_HOST_REG_PINNED` |
| `aclrtHostGetDevicePointer` | "**Obtains the device memory address** registered and mapped by `aclrtHostRegisterV2`"，签名 `(void *pHost, void **pDevice, uint32_t flag)`：输入 host 指针、输出 device 指针 | **命名易误读**:它不是"在 device 取 host 指针"，而是**host 侧的地址翻译器**——把 host 虚拟地址翻成 AI Core 能用的 device 地址,再作 kernel 入参(对位 CUDA `cudaHostGetDevicePointer`) |

目标使用流程：

```
# host 侧（CPU，准备阶段）
ptr = aclrtMallocHost(...) 或 malloc(...)            # 一块 host 内存（4K 页对齐）
aclrtHostRegisterV2(ptr, size, ACL_HOST_REG_MAPPED)  # 注册为 device 可访问
aclrtHostGetDevicePointer(ptr, &devPtr, 0)           # 翻译出 device 视角地址 devPtr
launch kernel(devPtr, ...)                           # devPtr 作为入参喂给算子
# device 侧（AI Core，kernel 内）
GlobalTensor.SetGlobalBuffer(devPtr); DataCopy(UB, gm, ...)  # ← 读的其实是那块 host 内存
```

仍需 P0b 实测的边界是两条注册路的官方措辞相反：

| 路 | 拿到的 devPtr | 官方原话 |
|--|--|--|
| **路 A**：`HostRegisterV2` + `HostGetDevicePointer` | 专门的映射地址 | "This address **cannot** be used for memory operations, such as memory copy." |
| **路 B**：`aclrtMallocHostWithCfg` + `ACL_RT_MEM_ATTR_VA_FLAG` / `vaFlag=1` | **与 host 地址相同**的统一地址 | "the mapped device address is the same as the host address and **can** be used for memory operations." |

"memory operations / memory copy" 是否涵盖 **kernel 内 `DataCopy`**，官方未明说。初步判断路 B（统一地址）更可能成立；但目标 A2/A3 是 EP 形态（PCIe/HCCS 分离），统一地址可用性和跨总线带宽必须上机测。P0b 影响见 §2.7.5。

> **官方文档 URL**（CANN Commercial 8.5.0，acl API C）：
> - [aclrtHostRegisterType 枚举](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1840.html)
> - [aclrtHostRegister](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1804.html) · [aclrtHostRegisterV2](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_2128.html)
> - [aclrtHostGetDevicePointer](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_2129.html)
> - [aclrtMallocHostWithCfg](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1799.html)

### 2.3 差异与对策

#### 2.3.1 swap-in 机制 → AscendC kernel 读 Host / staged DMA

GPU swap-in kernel 可以在一个 warp 内完成 hit/miss 判定和 Host KV 读取；NPU 需要把工作拆给不同单元：Vector 做批量比较，Scalar 做地址和调度，MTE 做 DMA 搬运。因此 NPU 实现应先稳定 hit/miss 判定，再决定 miss copy 放在 kernel 内还是退到 Host staged DMA。

| | GPU | NPU |
|--|--|--|
| Host 内存注册 | `cudaHostRegister` + UVA | `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer`（A3/A2 官方支持 device 读写，§2.2.2），或 `aclrtMallocHostWithCfg`+VA_FLAG 统一地址 |
| kernel 内读 Host | PTX `ld.global.nc`（texture cache streaming） | DataCopy via devPtr（能力官方背书；**P0b 仅实测 kernel 内 DataCopy 的具体可用性+带宽**：通过→零拷贝；不通过→退 staged DMA，§2.2.2） |
| 既有 host↔device 搬运范式 | kernel 内零拷贝直读 | 显式 staged DMA（`transfer_kv_dim_exchange` H2D/D2H，§2.2.1） |
| kernel 源码位置 | 本仓库 `hisparse.cuh` 可读可改 | 外部包 `sgl-kernel-npu`(已拉取 tag 2026.05.01),以 `csrc/cache_location_assign/` 为编码模板,需先打通构建(§2.2.1 / P0a) |
| 并行模型 | SIMT：32 lanes 各读不同地址，隐式同步 | SIMD：MTE DMA 搬整块，Scalar core 做判定 |
| Graph 兼容 | CUDA Graph 支持 | CANN Graph 待验证（路径②的 H2D 往返会打断单图捕获） |

算法逻辑仍是 hit 判定、LRU、miss 配对；实现瓶颈从 GPU scatter 转到 NPU hit 判定并行度。Scalar 只做收集/地址/派发，Vector 做批量比较，MTE 搬连续 token。完整算法见 §2.7。

#### 2.3.2 attention 寻址 → 把 logical token 翻译成 SFA 坐标

attention 算子按稀疏清单读取 KV。关键差异是清单里的数字属于哪个地址空间：GPU FlashMLA 的 `indices` 已经是物理 KV 行号，kernel 只做 `idx/page_size` 和 `idx%page_size`；当前 NPU GLM-5 SFA 的 `sparse_indices` 是 S2 逻辑位置，PA_BSND 分支还要查 `block_table` 才得到真实 KV block。

当前结论：**先走 PA_BSND `block_table` 重映射，再验证 BSND hot-slot。** 这是因为 GLM-5 生产调用已经是 `layout_query="TND"` + `layout_kv="PA_BSND"`，保留这套调用契约改动最小。BSND 一层直址确实存在，但它要求 `layout_query==layout_kv=="BSND"`，query/rope/output 形态都要跟着切。

| | GPU FlashMLA | NPU SFA |
|--|--|--|
| 寻址模式 | `indices` 是绝对物理扁平行号；kernel 不解引用 `block_table` | PA_BSND：`sparse_indices` + `block_table` 两级；BSND：`sparse_indices+s2StartIdx` 直接当 S 轴下标 |
| logical→physical 翻译 | kernel 外部已经算好 | PA_BSND 在 kernel 内查表；BSND 不绑定 `block_table` |
| 索引基准 | 绝对，无基址相加 | 窗口相对：`s2Idx = sparse_indices + s2StartIdx` |
| 当前 GLM-5 形态 | FlashMLA sparse decode 直接吃 `indices` | `layout_query="TND"`、`layout_kv="PA_BSND"`、带 `block_table` |
| 稀疏粒度 | token（page_size=1） | arch35（950 类）硬编 token-wise `sparse_block_size=1`；block-wise(2~128) 仅 arch22 |

| | GPU `flash_mla_with_kvcache` | NPU `npu_sparse_flash_attention` |
|--|--|--|
| Q | `q=q_input`，NoPE/RoPE 已拼在最后一维 | `query=q_nope` + `query_rope=q_pe`，当前 `layout_query="TND"` |
| KV | `k_cache=kv_cache`，单个 packed cache，FP8 sparse decode 要求 `[num_blocks, page_block_size, 1, bytes_per_token]` | `key=k_nope, value=k_nope, key_rope=k_pe`，当前 `layout_kv="PA_BSND"` |
| 稀疏清单 | `indices=page_table_1.unsqueeze(1)`，值已经是 FlashMLA-style encoded KV index | `sparse_indices=topk_indices`，值先作为 SFA sparse 清单，PA_BSND 下再经 `block_table` 翻译 |
| `block_table` | sparse decode 不解引用；SGLang 传空 tensor 只是为了 wrapper 不报错 | PA_BSND 必传，真实参与寻址 |
| 长度参数 | `cache_seqlens` / scheduler metadata 辅助调度；读哪些 token 由 `indices` 决定 | `actual_seq_lengths_query/kv` 参与 TND/PA_BSND 的 batch 边界和窗口边界 |

方案 A/B 不是字段改名，而是把 GPU 的物理 KV 行号语义落到 SFA 的 `key/value/key_rope + sparse_indices + block_table + actual_seq_lengths` 调用契约上。

**源码依据**：

- 对应算子是 `SparseFlashAttention`，不是 `SparseFlashMla`：[`op_host/sparse_flash_attention_def.cpp`](../ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp) 的输入/attr 列表与 `torch_npu.npu_sparse_flash_attention(...)` 调用点逐字段吻合。
- 两种寻址模式由 `isPa` 选：[`sparse_flash_attention_service_vector_mla.h:225-237`](../ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h#L225) 中，PA_BSND 走 `blockTableGm.GetValue(...)*blockSize + s2Idx%blockSize`；BSND 走 `realkeyOffset = boIdx*s2Size + s2Idx`。
- 索引是窗口相对的：`s2Idx = sparseIndicesGm.GetValue(...) + runInfo.s2StartIdx`，这是与 FlashMLA 绝对行号的真实差异。
- BSND 在 pytest/eager golden 有用例但只覆盖小 K：`bsnd_basic` 是 K=16，`bsnd_multi_batch` 是 K=32；K=2048 的 `pa_bsnd` golden 用例走 PA_BSND。op_host/tiling UT 里有 BSND+K=2048 的形状检查，但不能等价为 GLM-5 生产参数组合已覆盖。
- `return_softmax_lse=True` 时不支持图模式；HiSparse 当前用 False，不受限。BSND 还要求 `layout_query==layout_kv`。

**三条寻址方案**：

| 方案 | SFA 看到的 KV | 关键做法 | 代价/验证点 | 结论 |
|---|---|---|---|---|
| A：PA_BSND `block_table` 重映射 | 保持当前 `layout_query="TND"`、`layout_kv="PA_BSND"`、`block_table` | `block_table` 指向 hot buffer block；`sparse_indices` 可保留原始 logical S2，也可重写成 compact hot S2 | page 粒度驻留；block_size=128 时会有 page 放大；需 golden 量化 hot buffer 和 H2D 带宽 | 当前主线，改动最小 |
| B：BSND hot-slot 直读 | `key/value=[B,device_buffer_size,1,D]`，`block_table=None`，`layout_query="BSND"`，`layout_kv="BSND"` | `sparse_indices` 填 hot slot，例如 `407`；SFA BSND 分支直接读 `boIdx*s2Size+s2Idx` | query/rope/output 要从 TND+PA_BSND 切到 BSND；要处理 `s2StartIdx`；`actual_seq_lengths_kv` 要改成 hot buffer S2；K=2048 生产参数组合需自测 | 优化/备选方案 |
| C：改 SFA | 新 hot-slot 调用契约 | `sparse_indices` 直接表示 hot slot，或支持 FlashMLA-style encoded KV index | 要改 op host、tiling、kernel、测试和分发 | A/B 都对不齐 baseline 时最后备选 |

PA_BSND 方案 A 有两个变体：

| 变体 | 做法 | 代价 |
|---|---|---|
| A1：保留原始 logical S2 | `sparse_indices` 仍填原始长序列 logical token；`block_table` 覆盖原始逻辑块号空间，只是表项指向 hot buffer 物理块 | 表不能 compact；hot buffer 承受 page 放大 |
| A2：compact hot S2 | 把 `sparse_indices` 重写到小 hot 逻辑空间；`block_table` 只列 hot blocks | 每步要维护 logical token 到 compact token/page 的映射 |

PA_BSND 内部翻译不是缺陷：它减少 kernel launch，按 block 连续搬对 SIMD 友好。HiSparse 先复用这条分支做 `block_table` remap；BSND 用来验证 token-wise hot-slot 是否值得切调用契约。

**SFA 算子实际调用点**（`ascend_backend.py`，`forward_sparse` 路径）：

| 调用点 | 方法 | 触发条件 | 说明 |
|--------|------|---------|------|
| `ascend_backend.py:949` | `forward_sparse`（主路径） | 默认（非 CP / decode） | 单次 SFA，标准稀疏 attention |
| `ascend_backend.py:739` | `do_cp_balance_attn`（prev 半） | prefill + `is_nsa_enable_prefill_cp()` + `attn_cp_size > 1` | Context Parallel zigzag 切分 |
| `ascend_backend.py:761` | `do_cp_balance_attn`（next 半） | 同上 | 另一半 |

三处共用同一个 `torch_npu.npu_sparse_flash_attention` 算子，当前参数一致（`sparse_block_size=1`, `layout_kv="PA_BSND"`, `sparse_mode=3`, `attention_mode=2`, 都带 `block_table`）。方案 A/B 只改 Python 侧的调用参数：方案 A 保持 `PA_BSND` 但重映射 `block_table`；方案 B 把 `layout_kv` 切到 `"BSND"`、`block_table=None`、喂小 hot buffer 当 `key/value`、`sparse_indices` 填 hot slot。两者三处调用都能覆盖，且不碰算子源码；仅方案 C 需改算子内部寻址分支。

**CP 路径留到 P5**：`do_cp_balance_attn` 把序列切两半，每半有独立 `topk_indices_prev/next` 和 `actual_seq_lengths_kv`。HiSparse residency 状态按整序列管理，两个 SFA 调用如何共享 hot buffer、residency kernel 在 CP 下跑几次，需要单独 golden。

#### 2.3.3 内存池 → 新增 HiSparse 专用组件

不改现有 `NPUMLATokenToKVPool`，新增一套：

| GPU 组件 | NPU 对应（新增） | 职责 |
|----------|-----------------|------|
| `HiSparseNSATokenToKVPool` | `NPUHiSparseMLATokenToKVPool` | Device hot buffer，**布局由寻址方案定**：方案 A 用 PA_BSND + 重映射 block_table，方案 B 用 BSND（`sparse_indices` 直填 hot slot，§2.3.2）；只分配 `device_buffer_size` slots |
| `HiSparseTokenToKVPoolAllocator` | `NPUHiSparseAllocator` | logical↔device 映射，staging/decode 生命周期 |
| `MLATokenToKVPoolHost` + `cudaHostRegister` | `NPUMLATokenToKVPoolHost` + `aclrtHostRegisterV2`（§2.2.2） | Host pool，全量 KV，注册后 kernel 可直读 |

不动的：现有 pool/allocator、PA_BSND layout、`index_k_buffer`（常驻 NPU，不 offload）。

#### 2.3.4 Indexer KV 常驻

Indexer KV 维度小，全量常驻 NPU：

```
indexer_kv: 128K × 256 bytes × 61 layers = 1.95 GB（占 attention KV 的 ~22%）
attention_kv: 128K × 1152 bytes × 61 layers = 8.6 GB（需要 offload）
```

`npu_lightning_indexer` 调用不变。

### 2.4 请求生命周期（NPU 完整流程）

NPU 侧仍是 prefill → staging → decode → finish。下面先写 **PD 混合 / co-located** 形态；PD 分离下 D 节点跳过 staging，见 §2.4.1。

```
┌─────────────┐     ┌──────────┐     ┌──────────────────────┐     ┌────────┐
│   Prefill   │────>│ Staging  │────>│   Decode (per step)  │────>│ Finish │
└─────────────┘     └──────────┘     └──────────────────────┘     └────────┘
```

**Prefill**：

1. KV 写入 NPU 全量 pool（现有 `forward_dsa_prepare_npu` 路径不变）
2. Prefill 结束后，在 backup stream 上发起异步 D2H：`aclrtMemcpyAsync` 全量备份到 Host pool（`NPUMLATokenToKVPoolHost`）
3. `aclrtRecordEvent(prefill_done_event, backup_stream)`

**Staging**：

1. Poll `aclrtQueryEventStatus(prefill_done_event)` 等待 D2H 完成（或 `aclrtSynchronizeEvent` 阻塞等待）
2. 释放全量 pool 中该 request 的 slots（NPU 显存回收）
3. 分配 hot buffer：从 `NPUHiSparseMLATokenToKVPool` 分配 `device_buffer_size` 个 slots + 1 reserved slot
4. 初始化 per-request 状态：
   - `device_buffer_tokens[layer, req, slot]` 跟踪表（全 -1）
   - `lru_state[layer, req, :]` 初始化
   - `full_to_hisparse_device_mapping` 建立

**Decode（每步）**：

```
prepare_for_decode():
  │
  ├── seq_lens += 1
  ├── alloc_decode() → new out_cache_loc
  │
  └── map_last_loc_to_buffer():
        ├── backup_previous_token():
        │     在 backup_stream 上：
        │       aclrtStreamWaitEvent(backup_stream, main_stream_event)
        │       aclrtMemcpyAsync(host_pool[prev_pos], hot_buffer[prev_slot], D2H)
        │       aclrtRecordEvent(backup_done_event, backup_stream)
        │
        └── 映射 newest token → reserved slot (buffer_size 位置)

forward_decode() (per layer):
  │
  ├── main_stream wait backup_done_event
  │
  ├── 写 KV cache（newest token 写入 reserved slot）
  │
  ├── npu_lightning_indexer → top_k_tokens [B, 2048]
  │
  ├── AscendC residency kernel / judge kernel（详见 §2.3.1）:
  │     输入: top_k_tokens, host_kv_devPtr, hot_buffer, lru_state
  │     输出: hot_locations（A=hot block/page 映射；B=hot slot list）
  │
  ├── SFA 坐标转换层:
  │     输入: top_k_tokens, hot_locations
  │     输出: remapped sparse_indices / block_table
  │
  └── npu_sparse_flash_attention(           # 方案 A：保持 PA_BSND，不改算子
        query=q_nope, query_rope=q_pe,       # 保持当前 TND query
        key=hot_pa_cache, value=hot_pa_cache,
        key_rope=hot_pa_rope,
        sparse_indices=<原始 logical S2 或 compact hot S2>,
        block_table=<指向 hot buffer block 的 remapped table>,
        layout_query="TND", layout_kv="PA_BSND", sparse_block_size=1)
```

**Finish**：

1. 释放 Host pool 中该 request 的 slots
2. 释放 hot buffer slots（归还 `NPUHiSparseAllocator`）
3. 清理 per-req 跟踪状态（device_buffer_tokens, lru_state, mapping）

**与现有路径的修改点汇总**：

| 组件 | 现有行为 | HiSparse 后 |
|------|---------|-------------|
| KV 写入 | 写全量 pool | 写 hot buffer reserved slot + 异步 D2H backup |
| Indexer | `npu_lightning_indexer` | 不变 |
| Attention 输入 | 全量 kv_cache + sparse_indices + block_table | hot_buffer + SFA 坐标转换层产出的 `sparse_indices` / `block_table` |
| 新增 | — | AscendC residency kernel + backup stream 同步 |

#### 2.4.1 PD 混合 vs PD 分离的生命周期差异

§2.4 描述的是 **PD 混合**场景。**PD 分离**（prefill 在 P 节点、decode 在 D 节点）下，D 节点的生命周期**没有 Staging 阶段**。

根因：Staging 阶段的全部工作是“把 device 上的全量 KV 备份到 Host”（D2H copy）。PD 混合下 KV 是本地 prefill 算出来的，先在 device 上，必须自己搬；PD 分离下 KV 由 P 节点经 RDMA **直接写进 D 节点的 Host pool**，D 节点的 device 上从未出现过全量 KV，所以无备份可做、无 device 余量可释放。

| 阶段 | PD 混合（同一实例） | PD 分离（D 节点） |
|------|--------------------|-------------------|
| KV 来源 | 本地 prefill，全量常驻 device | P 节点 RDMA 直接写 D 节点 Host pool |
| device 是否有过全量 KV | 有 | **无**（只分配 logical 索引 + Host 目标） |
| **Staging（D2H 备份）** | **有**——staging 的核心开销 | **跳过** |
| 释放 device 余量槽位 | 有 | **无**（未占用过） |
| hot buffer 初始状态 | prefill 已填满，标记为 resident | **空**——需修正 |
| 准入触发点 | prefill 输出处理后 | D 节点 KV 传输完成后 |

GPU 侧对应实现：PD 混合走 `admit_request_into_staging`（含 D2H 备份），PD 分离走 `admit_request_direct`（跳过备份）。

**PD 分离的关键正确性修正**：`alloc_device_buffer` 默认把 hot buffer 标记为"已 resident"（`device_buffer_tokens = [0..buf_size-1]`），这在混合场景成立（prefill 真填满了），但 PD 分离下 buffer 是空的，必须修正：

- 短序列（seq_len ≤ device_buffer_size）：从 Host pool 预加载填满 buffer（`_preload_to_device_buffer`）
- 长序列：把 `device_buffer_tokens` 全置 -1，让 residency kernel 首步全 miss、从 Host 加载

漏掉此修正，decode 首步会读到 hot buffer 中的垃圾数据。

**NPU 适配影响**：

1. PD 分离需 P 节点把 KV 写入 D 节点经 `aclrtHostRegister` 注册的 Host pool（RDMA / `aclrtMemcpy`），D 节点准入路径跳过 D2H 备份、只建 hot buffer + 修正 resident 标记。
2. 优先做 PD 混合（§2.4），PD 分离作为后续场景（GPU 侧 dsv4 的 direct-to-host 路径目前仍是 `NotImplementedError`，仅 NSA 路径支持）。

#### 2.4.2 prefill 长度 vs hot buffer 大小的两种场景

设 `device_buffer_size = 4k`（hot buffer 容量）。staging 时 `alloc_device_buffer` 按 prefill 长度走两条分支（以下为 NSA 路径，`hisparse_coordinator.py:288-333` / `hisparse_memory_pool.py:240-276`）。

**两个尺寸常量**：

- `device_buffer_size`：hot buffer 标称容量（如 4k），decode 时 swap-in 的 LRU 缓存大小
- `padded_buffer_size = device_buffer_size + page_size`：实际分配量，多出的一页是写**当前新生成 token** 的 reserved slot（长序列时新 token 写在这页，不占 LRU 缓存区）

##### 场景 A：prefill ≤ 4k（短序列）

整个序列装得进 hot buffer，**无需 swap-in**。

| 步骤 | 行为 |
|------|------|
| staging 分配 | `alloc_size = 页对齐(prefill_len)`，**只分配实际需要的量**，不浪费到满 4k（`:297-300`） |
| device 留存 | 全部 prefill token 都留在 device，全 resident |
| decode | indexer 选 top-k → swap-in kernel 发现全命中，直接返回 device locs，**不读 Host** |
| 序列增长 | 每步 decode 若 `seq_len` 超过当前 buffer 容量，`_grow_device_buffers` 动态扩容（页对齐），直到触顶 `device_buffer_size`（`:335-406`，仅对 `seq_len ≤ device_buffer_size` 的 req 扩容） |
| 触顶 | 当 `seq_len` 达到 `device_buffer_size`，分配量提升到 `padded_buffer_size`（加上 reserved page），此后转入场景 B 的行为 |

PD 分离直连场景下短序列还需 `_preload_to_device_buffer` 从 Host 预热填满 buffer（见 §2.4.1）。

##### 场景 B：prefill > 4k（长序列）

序列装不下，**必须 swap-in**，hot buffer 当 LRU 缓存用。

| 步骤 | 行为 |
|------|------|
| staging 分配 | `alloc_size = padded_buffer_size`（4k + reserved page，`:301-302`） |
| device 留存 | 从 prefill 占用的全量槽位中**保留前 `device_buffer_size`（4k）个**作初始内容，其余 `free_hisparse_indices` 释放（`hisparse_memory_pool.py:250-251`） |
| 初始标记 | `device_buffer_tokens = [0,1,...,4095]`（`:328-330`），告诉 swap-in kernel buffer 初始装的是 token 0~4095 |
| decode | 每步 indexer 选 top-k → residency/judge kernel 判定 hit/miss → miss 从 Host 搬入、按 LRU 踢出旧 token → 输出 hot_locations |
| 新 token | 每步生成的新 token 写入 reserved slot（`device_buffer_size` 位置，`map_last_loc_to_buffer:458-460`），并异步增量备份到 Host（§2.4 decode 流程） |

**初始保留"前 4k"而非"最重要 4k"**：staging 时 decode 尚未开始、没有 query，无法判断 token 重要性，故按序列顺序取前 4k。这只是 LRU 缓存的初始值，decode 第一步 indexer 即按真实 query 选 top-k，miss 的从 Host 换入——一两步内 buffer 内容就被实际需要的 token 替换。

**冷启动代价**：第一步 decode 的 top-k 与"前 4k"初始内容大概率不重合（尤其序列尾部 token，局部性最强却不在 buffer），首步 miss 较多、有一批 Host 搬运。因相邻 decode 步 top-k 高度重叠，一两步后命中率回升至稳态。NPU 上 Host 读延迟若偏大，首步开销会被放大，可考虑 staging seeding 优化（保留尾部 recent + 头部 sink 混合，而不是只保留前 4k），见 §2.5。

### 2.5 性能风险

风险分两类：**增量风险**（HiSparse 新引入）和**存量风险**（SFA/现有路径本来就有，HiSparse 基本继承）。

**增量风险（HiSparse 新引入，按优先级）**：

| 风险 | 影响 | 缓解 / 实测项 |
|------|------|------|
| **Host 读带宽/延迟（主要风险）** | EP 形态下每个 miss 跨 PCIe/HCCS 搬 KV；带宽不够则 miss copy 开销吃掉稀疏化收益 | **P0b 必测带宽**；判定平面不依赖它（§2.7.5）；MTE 批量/double-buffer 掩盖 |
| HBM 散搬 miss | 公开 Ascend C 接口无"GM 按索引表批量 gather"，散搬只能逐段 `DataCopy`（§2.7.1） | 排序后合并连续段（`lightning_indexer` 现成向量排序）；或粗化到 block 粒度 |
| 长序列冷启动 | 首步 buffer 初始"前 4k"与真实 top-k 不重合、miss 多（§2.4.2）；Host 读延迟大时放大 | staging seeding：保留尾部 recent + 头部 sink 混合；或 staging 末尾预热 |
| 小池子 PA_BSND remap / BSND hot-slot 是否被 SFA 接受 | 方案 A/B 成立的前提（§2.3.2）。A 保持当前 GLM-5 `TND+PA_BSND+block_table` 调用契约，但要证明 remapped `block_table` 能让 SFA 读 hot buffer，且 page 放大后的 hot buffer/H2D 成本仍划算。B 的 BSND 一级直址分支确实存在：`realkeyOffset = boIdx*s2Size + s2Idx`（`service_vector_mla.h:233`），但 pytest/eager golden 的 BSND 用例仅 `K=16/32`，`K=2048` golden 走 PA_BSND；op_host/tiling UT 的 BSND+K=2048 形状检查不能替代生产形态验证。生产形态还要验证 query/rope reshape、`s2StartIdx`、`actual_seq_lengths_kv` | **P1 上机 golden 先测 A，再测 B**；生产参数组合：B 多 batch、decode S1=1、K=2048、BF16/FP16、query_rope/key_rope、actual lengths、s2StartIdx |

**存量风险（SFA 本就承担，非 HiSparse 新增）**：

| 项 | 说明 |
|------|------|
| SFA 离散访存 | 现有 SFA 已在 `sparse_block_size=1` 下散读 2048 token，算子文档 + 内核源码表明内部已做"指令缩减 + 搬运聚合"优化。**HiSparse 把散读范围从全量 KV 缩到 hot buffer，不会更差**；此前把它列为 HiSparse 新风险是误判（§2.3.2） |
| hit 判定并行度 | 纯 Scalar 线性扫 4096 slot 慢；必须 Vector 向量化（排序求交，§2.7.3.1），这是新 residency kernel 自身的实现要求 |

### 2.6 分阶段落地

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0a: `sgl-kernel-npu` 贡献链路** | 在已拉取的 `../sgl-kernel-npu/` 跑通 `build.sh`，以 `csrc/helloworld/` 为模板新增并加载一个自定义 AscendC 算子（本仓库无源码，§2.2.1） | helloworld 算子编译通过 + `torch.ops.npu` 可调用 + `tests/` 单测过 |
| **P0b: host 直读探针** | kernel 内 DataCopy 能否直读 registered host + 带宽 —— 决定搬运并进 kernel（①）还是退 staged DMA（②）；**非全局门控**，判定平面不依赖它（§2.7.5） | 最小 kernel + 带宽数据 + 路径结论 |
| **P1: SFA 寻址探针** | 先验证方案 A：保持 `layout_query="TND"`、`layout_kv="PA_BSND"`，用 remapped `block_table` 读 hot buffer，并二选一确定 A1 大表 / A2 compact。再验证方案 B：BSND hot-slot 直读，落定 query/rope 转 BSND、`s2StartIdx` 扣减、`actual_seq_lengths_kv` 改义。pytest/eager golden 仅覆盖 BSND K=16/32，生产形态须自测 | 单层 golden test：方案 A/B 与全量常驻 baseline 对齐 |
| **P2: AscendC residency kernel** | hit/miss/LRU/copy 全流程（路径②则为 judge-only kernel + host 侧 staged DMA） | 对齐 Python 参考实现 |
| **P3: 寻址方案定稿** | P1 通过则用方案 A 或 B（不改算子）做多层 decode 对齐；P1 不通过才退方案 C（改 SFA 内部寻址契约） | 多层 decode 对齐全量常驻 baseline |
| **P4: CANN Graph 验证** | 端到端 Graph capture/replay（路径② Graph 会被 H2D 往返打断，需评估） | 性能 benchmark |
| **P5: 生产化** | 多请求、生命周期、异常回收、CP 路径（`do_cp_balance_attn`） | 压力测试 |

P0a（外部包贡献链路）是 P2/P3 写算子的前提，必须先做。P0b 决定 P2 的 kernel 形态（路径①在 kernel 内完成判定+搬运，路径② judge-only + staged DMA），但**判定平面可不等 P0b 先行开写**（§2.7.5）：P0b 通过则把搬运并入，不通过则保持 judge-only。

### 2.7 AscendC Residency Kernel 算法设计

本节把 §2.3.1 的 GPU/NPU 执行差异落到可编码粒度。伪代码目标是向 `sgl-kernel-npu` 新增算子；路径①包含 Phase 2 的 kernel 内 copy，路径②把 Phase 2 移到 Host staged DMA。

核心约束只有两条：Scalar 弱，hit 判定必须交给 Vector；`DataCopy` 只能搬规则块，miss copy 要逐块发并用 double buffer overlap。

#### 2.7.1 硬件执行模型

AI Core（Atlas A3/A2）的相关执行单元：

| 单元 | 角色 | 官方描述 |
|------|------|----------|
| Scalar Unit | 地址计算、循环控制、分支判定、向其他 queue 派发指令 | "mini CPU"，计算能力弱，官方建议尽量减少 Scalar 工作量 |
| Vector Unit | SIMD 向量计算（批量比较、掩码） | 高吞吐并行计算 |
| MTE2 | GM → UB 搬运 | DMA 引擎 |
| MTE3 | UB → GM 搬运 | DMA 引擎 |

5 条独立指令队列（Vector / Cube / MTE1 / MTE2 / MTE3），**同 queue 内顺序执行，跨 queue 并行**，用 `SetFlag` / `WaitFlag` 做跨 queue 同步。Atlas A3/A2 是 Separation mode：Vector core 和 Cube core 分离，各有独立 Scalar Unit。

**关键能力边界**：`DataCopy` 只支持规则 stride（`DataCopyParams{blockCount, blockLen, srcStride, dstStride}`）和多维 slice（`SliceInfo`，最多 8 维），**不支持传入 indices 数组做任意地址 scatter/gather**。这直接决定了 miss copy 的实现形态（见 §2.7.3 Phase 2）。

#### 2.7.2 与 GPU kernel 的对应关系

| GPU swap-in kernel（warp 内） | NPU residency kernel（分 pipe） |
|------|------|
| 32 lanes 并行 probe hash table | Vector Unit 批量比较 + Scalar 收集结果 |
| lane 各自 `ld.global.nc` scatter read host | Scalar 循环派发 + MTE2 逐 token DataCopy |
| warp 内隐式同步 | SetFlag/WaitFlag 跨 queue 显式同步 |
| 一个 warp 同时做判定 + 搬运 | 判定（Vector/Scalar）与搬运（MTE）分离到不同 pipe，靠 double buffering overlap |

#### 2.7.3 算法伪代码（分 pipe）

> **代码依据**：下面的三阶段写法来自外部包里现成的 AscendC 算子 [`cache_loc_assign_kernel.cpp`](../sgl-kernel-npu/csrc/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp)（§2.2.1 (5)）。它已经示范了同款骨架：`GetBlockIdx()` 做 per-core tiling、`TPipe`+`TBuf`/`TQue` 手工管 UB、`DataCopy`/`DataCopyPad` 走 MTE、`SetFlag/WaitFlag<HardEvent::MTE2_V / V_MTE3>` 卡流水级。新 kernel 可直接以它为模板。它也暴露出逐元素 `GetValue/SetValue` 会落到 Scalar 循环；所以 Phase 1 要用 Vector 批量比较，Phase 2 要用 MTE 搬整块。

**Phase 1 — Hit/Miss 判定（Vector 主导）**

```
# Scalar: 加载跟踪表和请求
DataCopy(UB.device_buffer_tokens ← GM.device_buffer_tokens)  # MTE2, [buffer_size]
DataCopy(UB.top_k_tokens ← GM.top_k_tokens)                  # MTE2, [K]

# Vector: 批量判定每个 top_k token 是否已在 hot buffer
for each top_k token t (向量化，非标量循环):
    Compare(UB.hit_mask ← UB.device_buffer_tokens, t)        # Vector，批量比较 buffer_size 个 slot
    hit_slot = first set bit in hit_mask                     # 命中则记录 physical slot

# Scalar: 收集结果
miss_list   = [t for t in top_k if not hit]
evict_list  = LRU 选出 miss_list.size 个可驱逐 slot
更新 lru_state（hit 的刷新时间戳，evict 的标记）
```

**Phase 2 — Miss Copy（Scalar 派发 + MTE pipeline）**

```
# 无法一次 scatter，逐 miss 发射，double buffer overlap
InitBuffer(copy_que, 2, token_size)   # double buffering

for i, (miss_tok, evict_slot) in enumerate(zip(miss_list, evict_list)):
    src = host_kv_devPtr + miss_tok.pos * token_stride       # Scalar 算地址
    dst = hot_buffer     + evict_slot   * token_stride
    ub  = copy_que[i % 2]
    DataCopy(ub  ← src, token_size)    # MTE2: Host GM → UB
    DataCopy(dst ← ub,  token_size)    # MTE3: UB → hot buffer GM
    更新 device_buffer_tokens[evict_slot] = miss_tok
# 第 i 个 miss 的 MTE3 写出与第 i+1 个的 MTE2 读入并行
```

**Phase 3 — 输出**

```
for each top_k token t:
    hot_locations[t] = hit_location if hit else assigned_evict_location
DataCopy(GM.hot_locations ← UB.hot_locations)        # MTE3
```

##### 2.7.3.1 三个容易退化为 Scalar 的原语，以及替代写法

GPU 用的 warp 原语（hash 表 probe、ballot、shfl）在 NPU 上没有对应，照搬会退化成 Scalar 逐元素循环（§2.2.1 (5) 已看到这种写法）。替代思路是把随机访问改成规则向量计算：排序、比较、掩码收集、归约。下列 API 都能在已拉取的 `sgl-kernel-npu` 算子里找到真实调用：

| GPU 原语 | NPU 替代写法 | 代码依据（sgl-kernel-npu） |
|--|--|--|
| **hit 判定**：hash 表随机 probe | **排序 + 向量化求交**：驻留表 token-id 保持有序、top_k 排序，做有序集合 merge intersection（规则比较，不做随机 scatter） | `lightning_indexer` 有完整向量化全排序 [`lightning_indexer_vector.h:183-257`](../sgl-kernel-npu/csrc/lightning_indexer/op_kernel/lightning_indexer_vector.h#L183)：`AscendC::Sort32` + `MrgSort` 多级归并，注释明确支持 **`[128,256,384,512,1024,2048]` 个元素的 value+index 配对排序**，正好是 top_k≈2048 规模 |
| **miss 收集 / scatter**：warp 压缩出紧凑 miss_list | **`CompareScalar` 产掩码 + `GatherMask` 向量化收集**（消灭逐元素 SetValue） | DeepEP MoE dispatch 大量在用：`CompareScalar(..., CMPMODE::EQ/LT, ...)` + `GatherMask`（[`moe_distribute_dispatch_v2_single.h:728`](../sgl-kernel-npu/csrc/deepep/ops2/op_kernel/moe_distribute_dispatch_v2_single.h#L728)） |
| **LRU 驱逐**：ballot/shfl 并行选最久未用槽 | **时间戳 + 向量化 `argmin`/归约** | `WholeReduceMax`/`BlockReduceMax`/`ReduceMax` 在 `mla_preprocess`、DeepEP 多处在用（[`simd.h:82`](../sgl-kernel-npu/csrc/mla_preprocess/op_kernel/kernel/simd.h#L82)） |

**判定平面不依赖 host 直读**：排序求交判定、`GatherMask` 收集 miss_list、归约选驱逐槽都只访问 device 上的 `top_k_tokens/device_buffer_tokens/lru_state`。即使 P0b 证明 kernel 内不能高效读 Host，也能先做 judge-only kernel，把 miss_list 交给 Host staged DMA。含义见 §2.7.5。

> **粗化粒度（备选优化）**：GPU 是 token/页级驻留，scatter 压力大但 warp 可并行；NPU 更偏好规则块搬运。可把驻留粒度从单 token 粗化到“块”（一次换入连续一段），用一点缓存精度换更规则的判定和 DMA。这个取舍在 GPU 上不一定划算，在 NPU 上要重新测。落地形态正是方案 A（PA_BSND + 重映射 block_table，页粒度驻留，§2.3.2）。

#### 2.7.4 UB 内存分配

| buffer | 大小（buffer_size=4096, K=2048, BF16） | 说明 |
|--------|------|------|
| device_buffer_tokens | 4096 × 4B = 16 KB | 跟踪表（token id, int32） |
| top_k_tokens | 2048 × 4B = 8 KB | 本步请求 |
| hit_mask / 中间向量 | ~数 KB | Vector 比较中间结果 |
| copy double buffer | 2 × token_size（token_stride ≈ 1152B）≈ 2.3 KB | miss copy 暂存 |

总占用远小于 UB 容量（192 KB+），double buffer 可放大到更多级以加深 pipeline。

#### 2.7.5 性能分析

| 环节 | NPU | vs GPU |
|------|-----|--------|
| hit 判定 | Vector 批量比较 buffer_size 个 slot，向量化 | GPU warp 并行 probe。**Vector 是必须而非可选**：官方明确 Scalar 计算弱；纯 Scalar 线性扫描 4096 slot 是瓶颈 |
| miss copy | 逐 token DataCopy + double buffer overlap | GPU PTX streaming scatter。NPU 搬连续整 token（DMA 强项），**效率可能持平或更优**，但受限于"无法一次 scatter 所有 miss" |
| 地址计算 | Scalar 逐 miss 算 src/dst | GPU lane 内算。miss 数多时 Scalar 派发开销累积 |

结论：miss copy 不是瓶颈（DMA 搬连续块是 NPU 强项），**hit 判定必须走 Vector 向量化**，Scalar 仅做派发和收集。

**架构判断（结合 §2.2.2 与 §2.7.3.1）**：判定平面只依赖 device 上已有的向量 API，不依赖 host 直读，因此它本身就是一个完整的路径②判定实现：判定全程在 device kernel 内，只把紧凑 miss_list 交给 host 发 staged DMA。**这条判定平面不必等 P0b 就能开写。** P0b 因此不是总开关，而是决定是否把搬运并入 kernel：通过则走路径①，Graph 更友好；不通过则走路径②，多一趟往返，Graph 需分段。**注意路径②的搬运侧并非全现成**：BSND/token hot buffer 的 staged copy API 需新建（既有 `transfer_kv_dim_exchange` 仅适配 PA/page，§2.2）。

#### 2.7.6 参考文档

1. [CANN 8.5 Basic Architecture（AI Core 硬件架构）](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0007.html)
2. [CANN 8.5 Abstract Hardware Architecture（编程模型抽象）](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
3. [CANN 8.0 Control Units（指令队列与同步）](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
4. [Double Buffering Best Practice](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0033.html)
5. [TPipe/TQue Programming Paradigm](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00033.html)
6. [Basic DataCopy API（strided copy）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0103.html)
7. [Slice DataCopy API（多维 slice）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0105.html)

### 2.8 NPU Stream / Event 同步模型

搬运与计算分 stream 并行，用 event 保证顺序。唯一不能破的安全规则：Host 还没存好某 token 的副本前，hot buffer 对应位置不能被覆写。

#### 2.8.1 CANN stream/event 对应

| CUDA | CANN | 用途 |
|------|------|------|
| `cudaStream_t` | `aclrtStream` | 异步执行队列 |
| `cudaEvent_t` | `aclrtEvent` | 跨 stream 同步点 |
| `cudaStreamCreate` | `aclrtCreateStream` | 创建 stream |
| `cudaEventCreate` | `aclrtCreateEvent` | 创建 event |
| `cudaStreamWaitEvent` | `aclrtStreamWaitEvent` | stream 等待 event（非阻塞 host，stream 内排队等待） |
| `cudaEventRecord` | `aclrtRecordEvent` | 在 stream 上记录 event |
| `cudaEventSynchronize` | `aclrtSynchronizeEvent` | host 阻塞等待 event 完成 |
| `cudaEventQuery` | `aclrtQueryEventStatus` | host 非阻塞查询 event 状态 |
| `cudaStreamSynchronize` | `aclrtSynchronizeStream` | host 阻塞等待 stream 全部完成 |
| `cudaMemcpyAsync` | `aclrtMemcpyAsync`（`ACL_MEMCPY_DEVICE_TO_HOST`） | 异步 D2H / H2D copy |

API 名称与语义依据 AscendCL acl API (C) 官方文档核实（链接见 §2.8.4）。

HiSparse 用两条 stream：`main_stream`（forward decode）和 `backup_stream`（D2H 备份上一步 token）。

#### 2.8.2 时序（对标 doc01 §2.6 GPU 时序）

```
main_stream    │ write KV ──┬─ indexer ─ residency kernel ─ SFA ─ ... (step N)
               │            │
               │     record main_evt
               │            │
backup_stream  │   wait main_evt ─ D2H copy(prev token) ─ record backup_evt
               │                                              │
main_stream    │ ... (step N+1) wait backup_evt ─ write KV ───┘ (覆写 prev slot 前必须等 backup 完成)
```

关键约束：**main stream 覆写某 slot 前，该 slot 上一 token 的 D2H 备份必须已完成**——否则全量 KV 在 Host 上会丢失尚未备份的 token。靠 `backup_evt` 在 step N+1 的 KV 写入前对 main_stream 设 `aclrtStreamWaitEvent` 保证。

#### 2.8.3 与 GPU 模型的差异

| 维度 | GPU | NPU |
|------|-----|-----|
| Graph capture | CUDA Graph 天然支持含 host read 的 kernel | **SFA 本身官方明确支持图模式**（`return_softmax_lse=False` 时，§2.3.2）；但**自研 residency kernel 若含 registered-host devPtr 读**（路径①），其可 capture 性 **待 P4 验证**（路径②的判定 kernel 不含 host read，进图无此风险） |
| host read 在 stream 内行为 | UVA + PTX streaming，无显式同步 | devPtr DataCopy 走 MTE，是否需额外 barrier 待 P0b 验证 |
| 多 stream 调度 | SM 级抢占 | AI Core 调度粒度待确认，backup_stream 可能与 main_stream 争抢 MTE |

P0 探针（§2.6）需顺带验证 backup_stream 的 D2H 与 main_stream 的 residency kernel 在 MTE 上是否互相阻塞。

#### 2.8.4 参考文档

1. [AscendCL Synchronous Wait（aclrtSynchronizeStream / aclrtStreamWaitEvent / aclrtSynchronizeEvent）](https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000013.html)
2. [aclrtCreateStream（Stream Management）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0065.html)
3. [aclrtRecordEvent（Event Management）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0083.html)
4. [aclrtMemcpyAsync（Memory Management，含 `ACL_MEMCPY_DEVICE_TO_HOST`）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0106.html)
5. [acl API (C) API List（Atlas A2，含 event/stream 完整列表）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0014.html)

### 2.9 验收标准

1. **正确性**：HiSparse 输出 vs 全量常驻 baseline 输出的 logits max diff < 1e-3
2. **资源**：NPU KV 显存 ∝ O(device_buffer_size) 而非 O(max_seq_len)
3. **性能**：拆出 indexer / residency kernel / SFA / backup 四段耗时

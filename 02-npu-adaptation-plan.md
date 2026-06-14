## 第二部分：NPU HiSparse 适配方案

### 2.1 现状与目标

**当前 NPU DSA 路径**（`forward_dsa_prepare_npu` + `forward_dsa_core_npu`）已有完整的 sparse attention 推理：`npu_lightning_indexer` 选 top-k → `npu_sparse_flash_attention` 做 token 级 sparse attention。但 KV cache 全量常驻 NPU（PA_BSND layout, block_size=128），只省计算量不省显存。

| 维度 | CUDA DSA | NPU DSA |
|------|----------|---------|
| Indexer | FP8 paged MQA logits + fast_topk | `npu_lightning_indexer` |
| Attention | FlashMLA sparse / tilelang / FA3 | `npu_sparse_flash_attention` |
| KV layout | flat token pool, page_size=1 | PA_BSND, block_size=128 |
| HiSparse | 有（swap-in kernel） | **无** |
| KV 常驻 | 启用 HiSparse 时不常驻 | 全量常驻 |

**核心缺失**：没有 residency runtime（hit/miss/LRU/swap-in）。`aclrtHostRegister` 提供了 Device 访问 Host 内存的能力（Atlas A3/A2），但当前路径未使用。

**目标**：KV cache 不全量常驻 NPU，只保留 hot buffer，全量 KV 放 host，按需 swap-in。

### 2.2 主线方案

架构 1:1 对标 GPU HiSparse：通过 `aclrtHostRegister` 让 AscendC kernel 直接读取 Host KV，在 kernel 内原子完成 hit/miss/LRU/copy，输出 physical hot slots 给 SFA。

```
每步 decode (per layer):
  npu_lightning_indexer → top_k_tokens [B, 2048]
  → AscendC residency kernel:
      输入: top_k_tokens, host_kv_devPtr, hot_buffer, lru_state
      kernel 内部: hit/miss 判定 → LRU 更新 → miss DataCopy Host→hot_buffer
      输出: top_k_device_locs [B, K]
  → npu_sparse_flash_attention(
      key=hot_buffer, value=hot_buffer,
      sparse_indices=top_k_device_locs,
      addressing_mode=PHYSICAL_HOT_SLOT)
```

**待验证前提**（P0 门控）：

1. `aclrtHostRegister` 返回的 `devPtr` 能否在 AscendC kernel 中做 DataCopy 读取
2. 包含 Host memory read 的 AscendC kernel 能否被 CANN Graph capture/replay
3. Host memory read 延迟是否可接受

如果探针 1 不通过，退化为 Host 侧显式 H2D copy（stream + event），会打断 Graph 但仍可保证正确性。

### 2.3 差异与对策

#### 2.3.1 swap-in 机制 → AscendC kernel 直读 Host

| | GPU | NPU |
|--|--|--|
| Host 内存注册 | `cudaHostRegister` + UVA | `aclrtHostRegister` (Atlas A3/A2) |
| kernel 内读 Host | PTX `ld.global.nc`（texture cache streaming） | DataCopy via devPtr（待验证） |
| 并行模型 | SIMT：32 lanes 各读不同地址，隐式同步 | SIMD：MTE DMA 搬整块，Scalar core 做判定 |
| Graph 兼容 | CUDA Graph 天然支持 | CANN Graph 待验证 |

**SIMD vs SIMT 实现差异**：

GPU 用一个 warp 同时处理"判定"和"搬运"（32 lanes scatter/gather）。NPU 上这两件事分离到不同 pipe：

- **Scalar core**：miss list 收集、LRU 更新、地址计算、向 MTE 派发
- **MTE**：DataCopy 搬运 Host→UB→hot_buffer（DMA 引擎搬连续块，天然适合整 token 搬运）
- **Vector core**：向量化批量比较做 hit 判定（**必须**，非可选——Scalar 计算弱，详见 §2.7.5）

算法逻辑照搬 GPU（hit 判定、LRU、miss 配对），实现从"warp 内 SIMT 协作"变为"Vector 判定 + Scalar 派发 + MTE 搬运 pipeline"。miss copy 效率不会比 GPU 差（DMA 搬连续块是 NPU 强项），瓶颈在 hit 判定的并行度。完整算法设计见 §2.7。

#### 2.3.2 attention 寻址 → 扩展 SFA 支持 physical hot slot

| | GPU FlashMLA | NPU SFA |
|--|--|--|
| 寻址模式 | `indices` = physical offset（唯一模式） | `sparse_indices`(logical) + `block_table`(logical→PA physical) 两层跳转 |
| logical→physical 翻译 | kernel **外部**完成 | kernel **内部**融合 |
| HiSparse 需要改 attention 吗 | 不需要 | 需要（加一条跳过翻译的分支） |

**为什么 NPU SFA 把翻译融进 kernel**：

1. 减少 kernel launch（NPU launch 比 GPU 贵）
2. DMA 紧耦合（查 block_table 得到 page 地址后立刻 DMA 该 page，拆开多一次 Global Memory 往返）
3. 按 block 连续搬（128 tokens/次）对 SIMD/DMA 友好

**路线 C（推荐）**：给 SFA 新增 `PHYSICAL_HOT_SLOT` 寻址模式——`sparse_indices` 直接解释为 hot buffer 内的 physical slot offset，跳过 block_table 翻译，直接 `hot_buffer_ptr + indices[i] * stride` 读 KV。改动集中在 SFA kernel 的 `DataCopyPA` / `GetKeyGmOffset` 分支。

如果路线 C 的 scatter read 性能不达标（见 § 2.5），**路线 B 作为备选**：每步把 top-k token gather 到 compact PA buffer，让 SFA 连续读。牺牲 LRU hit 缓存收益，但保留 SFA 连续读性能。

**SFA 算子实际调用点**（`ascend_backend.py`，`forward_sparse` 路径）：

| 调用点 | 方法 | 触发条件 | 说明 |
|--------|------|---------|------|
| `ascend_backend.py:949` | `forward_sparse`（主路径） | 默认（非 CP / decode） | 单次 SFA，标准稀疏 attention |
| `ascend_backend.py:739` | `do_cp_balance_attn`（prev 半） | prefill + `is_nsa_enable_prefill_cp()` + `attn_cp_size > 1` | Context Parallel zigzag 切分 |
| `ascend_backend.py:761` | `do_cp_balance_attn`（next 半） | 同上 | 另一半 |

三处共用同一个 `torch_npu.npu_sparse_flash_attention` 算子，参数一致（`sparse_block_size=1`, `layout_kv="PA_BSND"`, `sparse_mode=3`, `attention_mode=2`, 都带 `block_table`）。**路线 C 改算子内部寻址分支，一次改动三处都受益**，不需改三遍 Python 调用。

**CP 路径需单独验证**：`do_cp_balance_attn` 把序列切两半，每半有独立的 `topk_indices_prev/next` 和 `actual_seq_lengths_kv`。HiSparse 的 hot buffer / residency 状态按整序列管理，切半后两个 SFA 调用如何共享 hot buffer、residency kernel 在 CP 下跑几次，是当前方案未覆盖的点。**第一阶段（P0–P3）聚焦非 CP 主路径（`:949`），CP 路径留到 P5**（见 §2.6）。

#### 2.3.3 内存池 → 新增 HiSparse 专用组件

不整改现有 `NPUMLATokenToKVPool`，新增一套：

| GPU 组件 | NPU 对应（新增） | 职责 |
|----------|-----------------|------|
| `HiSparseNSATokenToKVPool` | `NPUHiSparseMLATokenToKVPool` | Device hot buffer，PA_BSND layout，只分配 `device_buffer_size` slots |
| `HiSparseTokenToKVPoolAllocator` | `NPUHiSparseAllocator` | logical↔device 映射，staging/decode 生命周期 |
| `MLATokenToKVPoolHost` + `cudaHostRegister` | `NPUMLATokenToKVPoolHost` + `aclrtHostRegister` | Host pool，全量 KV，注册后 kernel 可直读 |

不动的：现有 pool/allocator、PA_BSND layout、`index_k_buffer`（常驻 NPU，不 offload）。

#### 2.3.4 Indexer KV 常驻

Indexer KV 维度小，全量常驻 NPU：

```
indexer_kv: 128K × 256 bytes × 61 layers = 1.95 GB（占 attention KV 的 ~22%）
attention_kv: 128K × 1152 bytes × 61 layers = 8.6 GB（需要 offload）
```

`npu_lightning_indexer` 调用不变。

### 2.4 请求生命周期（NPU 完整流程）

对标 GPU HiSparse 的 prefill → staging → decode → finish 四阶段，NPU 侧完整流程如下。**本节描述 PD 混合（prefill 与 decode 在同一实例）场景**；PD 分离场景的差异见 §2.4.1。

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
  ├── AscendC residency kernel（详见 §2.3.1）:
  │     输入: top_k_tokens, host_kv_devPtr, hot_buffer, lru_state
  │     输出: top_k_device_locs [B, K]
  │
  └── npu_sparse_flash_attention(
        key=hot_buffer, value=hot_buffer,
        sparse_indices=top_k_device_locs,
        addressing_mode=PHYSICAL_HOT_SLOT)
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
| Attention 输入 | 全量 kv_cache + sparse_indices + block_table | hot_buffer + top_k_device_locs |
| 新增 | — | AscendC residency kernel + backup stream 同步 |

#### 2.4.1 PD 混合 vs PD 分离的生命周期差异

§2.4 描述的是 **PD 混合**场景。**PD 分离**（prefill 在 P 节点、decode 在 D 节点）下，D 节点的生命周期**没有 Staging 阶段**。

根因：Staging 阶段的全部工作是"把 device 上的全量 KV 备份到 Host"（D2H copy）。PD 混合下 KV 是本地 prefill 算出来的、人在 device 上，必须自己搬；PD 分离下 KV 由 P 节点经 RDMA **直接写进 D 节点的 Host pool**，D 节点的 device 上从未出现过全量 KV，所以无备份可做、无 device 余量可释放。

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
| decode | indexer 选 top-k → swap-in kernel 走 fast path 直接返回 device locs，**不读 Host**（全命中） |
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
| decode | 每步 indexer 选 top-k → swap-in kernel 判定 hit/miss → miss 从 Host 搬入、按 LRU 踢出旧 token → 输出 `top_k_device_locs` |
| 新 token | 每步生成的新 token 写入 reserved slot（`device_buffer_size` 位置，`map_last_loc_to_buffer:458-460`），并异步增量备份到 Host（§2.4 decode 流程） |

**初始保留"前 4k"而非"最重要 4k"**：staging 时 decode 尚未开始、没有 query，无法判断 token 重要性，故按序列顺序取前 4k。这只是 LRU 缓存的初始值，decode 第一步 indexer 即按真实 query 选 top-k，miss 的从 Host 换入——一两步内 buffer 内容就被实际需要的 token 替换。

**冷启动代价**：第一步 decode 的 top-k 与"前 4k"初始内容大概率不重合（尤其序列尾部 token，局部性最强却不在 buffer），首步 miss 较多、有一批 Host 搬运。因相邻 decode 步 top-k 高度重叠，一两步后命中率回升至稳态。NPU 上 Host 读延迟若偏大，此首步 penalty 被放大，可考虑 staging seeding 优化（保留尾部/sink 混合而非纯前 4k），见 §2.5。

### 2.5 性能风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Host 读延迟高 | miss penalty 大，hit rate 对性能影响被放大 | P0 探针实测；MTE 批量搬运掩盖延迟 |
| SFA scatter read 慢 | 路线 C 按 2048 个散乱 slot 逐个读，比当前按 block 连续读慢 | 关注 MTE gather 模式；若不达标退回路线 B（compact gather） |
| hit 判定并行度 | 纯 Scalar 线性扫描 4096 slots 串行慢（官方建议 minimize scalar work） | Vector 向量化批量比较，Scalar 仅做收集/派发（详见 §2.7.5） |
| 长序列冷启动 | 首步 decode buffer 初始为"前 4k"，与真实 top-k 不重合，miss 多（§2.4.2）；NPU Host 读延迟大时首步 penalty 放大 | staging seeding 优化：保留尾部 recent + 头部 sink 混合而非纯前 4k；或 staging 末尾预热相关 token |

### 2.6 分阶段落地

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0: aclrtHostRegister 探针** | AscendC kernel 能否通过 devPtr 读 Host | 最小 kernel + 延迟数据 |
| **P1: SFA 寻址探针** | 定位 DataCopyPA / GetKeyGmOffset 改动点 | 单层 golden test |
| **P2: AscendC residency kernel** | hit/miss/LRU/copy 全流程 | 对齐 Python 参考实现 |
| **P3: SFA 路线 C 原型** | 扩展寻址 ABI | 多层 decode 对齐 full-resident |
| **P4: CANN Graph 验证** | 端到端 Graph capture/replay | 性能 benchmark |
| **P5: 生产化** | 多请求、生命周期、异常回收、CP 路径（`do_cp_balance_attn`） | 压力测试 |

P0 是门控节点。

### 2.7 AscendC Residency Kernel 算法设计

§2.3.1 给出了 SIMD vs SIMT 的概念差异，本节展开到可据此编码的粒度。**调研依据**：以下设计基于 CANN 官方文档对 AI Core 硬件架构、指令队列、DataCopy 能力的描述（链接见 §2.7.6）。

#### 2.7.1 硬件执行模型

AI Core（Atlas A3/A2）的相关执行单元：

| 单元 | 角色 | 官方描述 |
|------|------|----------|
| Scalar Unit | 地址计算、循环控制、分支判定、向其他 queue 派发指令 | "mini CPU"，计算能力弱，官方建议 minimize scalar work |
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

核心差异：GPU 把判定和搬运压在一个 warp 内，NPU 必须拆成"先批量判定、再 pipeline 搬运"两阶段。

#### 2.7.3 算法伪代码（分 pipe）

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
    top_k_device_locs[t] = hit_slot if hit else assigned_evict_slot
DataCopy(GM.top_k_device_locs ← UB.top_k_device_locs)        # MTE3
```

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
| hit 判定 | Vector 批量比较 buffer_size 个 slot，向量化 | GPU warp 并行 probe。**Vector 是必须而非可选**——官方明确 Scalar 计算弱、应 minimize；纯 Scalar 线性扫描 4096 slot 是瓶颈 |
| miss copy | 逐 token DataCopy + double buffer overlap | GPU PTX streaming scatter。NPU 搬连续整 token（DMA 强项），**效率可能持平或更优**，但受限于"无法一次 scatter 所有 miss" |
| 地址计算 | Scalar 逐 miss 算 src/dst | GPU lane 内算。miss 数多时 Scalar 派发开销累积 |

结论：miss copy 不是瓶颈（DMA 搬连续块是 NPU 强项），**hit 判定必须走 Vector 向量化**，Scalar 仅做派发和收集。这修正了 §2.3.1 把 Vector 写成"可选"的表述。

#### 2.7.6 参考文档

1. [CANN 8.5 Basic Architecture（AI Core 硬件架构）](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0007.html)
2. [CANN 8.5 Abstract Hardware Architecture（编程模型抽象）](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
3. [CANN 8.0 Control Units（指令队列与同步）](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
4. [Double Buffering Best Practice](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0033.html)
5. [TPipe/TQue Programming Paradigm](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00033.html)
6. [Basic DataCopy API（strided copy）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0103.html)
7. [Slice DataCopy API（多维 slice）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0105.html)

### 2.8 NPU Stream / Event 同步模型

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
| Graph capture | CUDA Graph 天然支持含 host read 的 kernel | CANN Graph 对含 `aclrtHostRegister` devPtr 读的 kernel 能否 capture **待 P4 验证** |
| host read 在 stream 内行为 | UVA + PTX streaming，无显式同步 | devPtr DataCopy 走 MTE，是否需额外 barrier 待 P0 验证 |
| 多 stream 调度 | SM 级抢占 | AI Core 调度粒度待确认，backup_stream 可能与 main_stream 争抢 MTE |

P0 探针（§2.6）需顺带验证 backup_stream 的 D2H 与 main_stream 的 residency kernel 在 MTE 上是否互相阻塞。

#### 2.8.4 参考文档

1. [AscendCL Synchronous Wait（aclrtSynchronizeStream / aclrtStreamWaitEvent / aclrtSynchronizeEvent）](https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000013.html)
2. [aclrtCreateStream（Stream Management）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0065.html)
3. [aclrtRecordEvent（Event Management）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0083.html)
4. [aclrtMemcpyAsync（Memory Management，含 `ACL_MEMCPY_DEVICE_TO_HOST`）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0106.html)
5. [acl API (C) API List（Atlas A2，含 event/stream 完整列表）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0014.html)

### 2.9 验收标准

1. **正确性**：HiSparse 输出 vs full-resident 输出的 logits max diff < 1e-3
2. **资源**：NPU KV 显存 ∝ O(top_k) 而非 O(max_seq_len)
3. **性能**：拆出 indexer / residency kernel / SFA / backup 四段耗时

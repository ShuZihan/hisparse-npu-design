## 第四部分：NPU HiSparse 适配方案分析

### 4.1 目标

在 SGLang Ascend NPU 路径上实现 HiSparse 的核心能力：**KV cache 不全量常驻 NPU，只保留 hot buffer，全量 KV 放 host，按需 swap-in。**

### 4.2 核心语义鸿沟

将 CUDA HiSparse 移植到 NPU 面临三个根本差异：

**差异 1：swap-in 机制**

| | CUDA HiSparse | NPU 迁移难点 / 可选方向 |
|---|---|---|
| 执行者 | GPU kernel 内部 | 若走 Host offload，copy 通常由 Host runtime 发起；若要求 Graph 不断，需要 NPU/AscendC op 或 Graph 内 copy 表达 |
| 原子性 | hit/miss/LRU/copy 一个 kernel 完成 | Host copy 会拆开原子路径；Graph-first 路线需要重新构造固定 shape residency op |
| 延迟 | 极低（kernel 内 warp-level DMA） | Host copy + stream 同步延迟高；device compressed backing 可把 miss 搬运变成 NPU 内 D2D/decompress |
| 固定 shape | 天然支持 CUDA graph | Explicit Host offload 会打断 graph；Graph-first 路线必须避免 Python per-step 分支 |

CUDA HiSparse 的单 kernel 方案需要先作为基准方案分析，而不能简单理解成“把几个步骤融合起来”。它真正有价值的地方是：在一个固定 shape 的 device kernel 内完成 residency update，避免 decode graph 在 attention 前被 Host 动态控制面打断。

GPU 方案的优点：
- **Graph 友好**：hit/miss/LRU/copy 都在 kernel 内，Python 不参与每步 miss plan，适合 CUDA graph replay。
- **原子性强**：metadata 更新、miss slot 选择、host → device KV 搬运和 `top_k_device_locs` 输出在同一个 kernel 中完成，attention 前状态一致。
- **同步少**：不需要把 top-k/miss plan 拉回 Host 再发起 H2D copy，避免 device → host → device 往返。
- **粒度精细**：token 级 swap-in，miss penalty 比 block 级换入更小。

GPU 方案的代价：
- **强依赖 CUDA 能力**：依赖 GPU kernel 直接访问 mapped pinned host memory，以及 CUDA/PTX 对这种访问路径的支持。
- **copy 隐藏在 kernel 内**：miss 多时 kernel 会等待远端 host memory 访问，copy 成本不容易和计算调度分离。
- **可移植性弱**：一旦换到 Ascend NPU，Host↔Device copy 的执行模型、Graph capture 能力、attention op 的寻址接口都不同。

因此 NPU 侧不应直接把“GPU fused kernel”当作唯一方案，而应按 Graph 约束重新分层：

| 方案 | 核心思路 | 优点 | 代价 | 判断 |
|------|----------|------|------|------|
| **方案 A：Graph-first fused residency op** | 在 NPU/AscendC op 内完成 top-k → hit/miss → LRU → 固定 shape resident table；若 CANN Graph 能表达 copy 节点或 device 侧可访问 Host backing，再把搬运也放进 graph | 最接近 CUDA HiSparse，保留 decode graph，性能上限最高 | 依赖 CANN Graph、Ascend runtime 和 Host memory access 能力验证 | **理想主线，但必须先做可行性探针** |
| **方案 B：Device compressed backing store** | 全量 KV 不放 Host，而是以 FP8/INT4/C4/压缩格式常驻 NPU；miss 时做 NPU 内 D2D copy/dequant/decompress 到 hot buffer | Graph 不断，所有 residency 工作都在 device 内，避免 Host copy 同步 | 显存节省少于 Host offload；量化/压缩会引入精度和模型适配风险 | **更 NPU-native，适合作为生产主线候选** |
| **方案 C：Explicit Host offload copy backend** | Host 侧生成 miss copy plan，用 stream/event 显式 H2D copy miss KV，再进入 attention | 语义最接近“全量 KV 在 Host”，容易做正确性 POC，显存节省最大 | 会打断 graph；top-k 到 Host 控制面可能引入同步和抖动 | **适合 POC/benchmark，不应在 Graph 是硬约束时作为最终主线** |

如果 **Graph capture 是硬约束**，NPU 适配目标应从“复刻 CUDA 的 host-backed swap-in kernel”调整为“保留 CUDA HiSparse 的 residency 语义，但让 residency update 可被 graph capture”。换句话说，优先验证方案 A；如果 Host backing 无法进入 graph，则优先考虑方案 B；方案 C 只能作为 correctness prototype 和性能下界测量。

**差异 2：sparse 粒度**

| | CUDA HiSparse | NPU npu_sparse_flash_attention |
|---|---|---|
| sparse attention 粒度 | token 级（page_size=1） | 由 `sparse_block_size` 决定；当前 SFA 路径 `sparse_block_size=1`，也是 token 级 |
| top_k 含义 | 2048 个 token | 2048 个 logical KV spans；当 `sparse_block_size=1` 时是 2048 个 token |
| KV 物理布局 | flat token pool / page_size=1 | PA_BSND paged cache，通常 PA block_size=128 |
| attention 输入 | `page_table = top_k_device_locs` | `sparse_indices + block_table + full kv_cache` |

这里必须区分两层“block”：

- **SFA sparse block**：`sparse_indices[i]` 选择 `[i * sparse_block_size, (i + 1) * sparse_block_size)` 这样的 logical KV span。当前 `sparse_block_size=1`，因此 SFA 原生语义是 token-level sparse attention。
- **PA cache block**：`block_table` 用于把 logical token/span 翻译到 PA_BSND 物理 KV cache 中的 page/offset。PA block_size=128 是存储分页粒度，不代表 SFA 一次必须选 128 个 token。

因此 NPU SFA 的问题不是“attention 只能 block 级稀疏”。它已经能在 `sparse_block_size=1` 下做到 token 级 sparse attention。真正缺失的是 HiSparse 的 **token 级 residency runtime**：哪些 top-k token 当前在 device hot buffer、miss token 如何换入、attention 如何只读取 hot buffer 中的 resident token。

#### 为什么 GPU HiSparse 不选择 block hot buffer + in-block token mask

HiSparse 的目标不只是“算得少”，而是 **搬得少、存得少、算得少** 三者同时做到 O(top_k)。

GPU 上的 HiSparse 方案是：完整 KV cache 放在 CPU pinned memory，GPU 上只维护一个小的 hot buffer，只存 top-k 选中的 token。每次 decode 时：

1. Indexer 打分，选出 top-k token
2. 不在 GPU 上的 token 从 CPU 搬到 hot buffer
3. attention 只读 hot buffer 里的 top-k token

如果按 PA block 粒度管理 hot buffer（整块搬、整块存），然后用 in-block mask 在 attention 时屏蔽 block 内未选中的 token，会变成：

| 维度 | token-level HiSparse | block hot buffer + in-block mask |
|------|----------------------|----------------------------------|
| GPU 上存多少 | top_k tokens | unique_blocks × block_size tokens |
| CPU → GPU 搬多少 | miss tokens | miss blocks × block_size tokens |
| attention 有效语义 | top_k tokens | mask 后仍是 top_k tokens |

举例：`top_k=2048`、`block_size=128`。如果 2048 个 token 分布在 800 个 block 里：

```
token-level HiSparse:
  存 / 搬 / 算 2048 tokens

block hot buffer + mask:
  存 / 搬 800 × 128 = 102400 tokens
  attention 语义通过 mask 回到 2048 tokens
```

mask 可以让 softmax 语义正确，但不能让 GPU 少存、少搬。显存占用和传输带宽——HiSparse 最想省的东西——会被 block 粒度显著放大。

其他补充原因：
- **block 内跳读也需要改 attention load 路径**：如果想在 block 内只 load 被 mask 选中的 token，本质上就是 token 级 gather，会破坏 block 连续访存优势，改造代价接近扩展 / 改写 attention kernel 的 indexing 与 load 分支。
- **metadata 更复杂**：token-level 只需要 `top_k_indices[B, K]` / `top_k_device_locs[B, K]`；block+mask 需要维护 `unique_blocks`、in-block bitmask、block→hot slot 映射和 offset 映射，对 graph capture 不友好。
- **LRU 缓存污染**：block 粒度 LRU 意味着一个热 token 会让同 block 的其他冷 token 也留在 hot buffer 里，挤掉真正有用的 hot tokens。

一句话：**block+mask 是语义补丁，让最终 attention 结果接近 token top-k；但它没有解决 residency/copy 的 O(top_k) 目标。GPU HiSparse 没有因此自研 attention kernel，而是复用已有 FlashMLA / FA3 / tilelang 的 sparse/paged attention 接口，并让 swap-in kernel 输出 physical hot slots，使 indexing、residency、copy、compute 全链路都在 token 粒度上对齐。**

**差异 3：attention kernel 消费方式**

| | CUDA (FlashMLA sparse) | NPU (npu_sparse_flash_attention) |
|---|---|---|
| KV 来源 | hot buffer（小 pool） | 全量 kv_cache（大 pool） |
| attention kernel | 复用已有 FlashMLA sparse / FlashMLA KV / FA3 / tilelang | 复用 `npu_sparse_flash_attention` |
| HiSparse 自定义部分 | `load_cache_to_device_buffer_kernel` 做 hit/miss/LRU/copy，输出 physical hot slots | 当前没有对应 residency op；SFA 也没有 direct physical hot slot 模式 |
| 寻址 | `indices` / `page_table` 类输入直接指向 k_cache / hot buffer 内的 physical locations | `sparse_indices` 给出 logical token/span，`block_table` 做 logical→PA physical page/offset 翻译 |
| 语义 | "只读 hot buffer 中这些 resident token" | "从全量 PA cache 中只算 sparse_indices 指定的 logical token/span" |

源码证据：

- `jit_kernel/csrc/hisparse.cuh` 定义的是 `load_cache_to_device_buffer_kernel`：扫描 top-k 与 LRU hot buffer，处理 hit/miss，把 miss KV 从 host cache 搬到 device buffer，并写出 `top_k_device_locs`。
- `srt/managers/hisparse_coordinator.py` 的 `swap_in_selected_pages()` 返回 `top_k_device_locs`；DSv4 路径在 `srt/layers/attention/dsv4/indexer.py` 中把返回值直接覆盖为 `core_metadata.c4_sparse_page_indices`。
- GPU 侧 FlashMLA 不需要修改的原因见 [01-gpu-hisparse-architecture.md § 2.4](01-gpu-hisparse-architecture.md#24-swap-in-机制核心-cuda-kernel)——核心是 hot buffer 就是 KV cache tensor 的子区域，`top_k_device_locs` 和 FlashMLA `indices` 的语义天然一致。
- NPU 路径里，`sfa_v1.py` 调用 `npu_sparse_flash_attention(... sparse_indices=topk_indices, sparse_block_size=1, block_table=block_table, layout_kv="PA_BSND")`；SFA kernel 内 `DataCopyPA` 会用 `s2Idx / blockSize` 查 `blockTableGm`，再从单一 `keyGm` / `kRopeGm` 指针按 offset 读。

传统 PagedAttention / PA `block_table` 能工作，有一个隐含前提：**所有 logical token 在 device 上都有对应的 physical KV slot**。`block_table` 像一张完整页表：

```
block_table[logical_block_id] -> device_physical_block_id
```

HiSparse 启用后，这个前提被破坏：

```
128K token 的完整 KV -> Host
GPU hot buffer        -> 4096 token slots
```

此时大多数 logical token 在 device 上没有 physical slot，不能再用一张完整 `block_table` 表达“所有 logical token 在 device 哪里”。HiSparse 需要表达的是当前 attention 工作集：

```
top_k logical tokens -> current physical hot slots
```

这就是 `top_k_device_locs`。

为什么 GPU HiSparse 不把传统 `block_table` 当作 hot-set 寻址输入：

1. **完整映射前提失效**
   传统 `block_table` 表达完整 logical KV address space；HiSparse 的 device resident set 只是完整序列的一个小子集。没有 resident 的 token 不能填 device address，表会变成大量 invalid entries 的残缺页表。

2. **映射关系每步变化**
   传统 PA 中，request 的 logical block 写入哪个 physical block 后基本保持不变，直到 request 结束。HiSparse hot slots 会被 LRU 反复复用：

   ```text
   step t:   hot_slot_0 = token_5,   hot_slot_1 = token_20
   step t+1: hot_slot_0 = token_5,   hot_slot_1 = token_300
   step t+2: hot_slot_0 = token_888, hot_slot_1 = token_300
   ```

   如果用 `block_table` 表达，需要每步清理 evicted token、写入 newly resident token。`top_k_device_locs[B, K]` 则由 swap-in kernel 每步直接输出，天然只描述本步有效工作集。

3. **粒度与布局不匹配**
   PA `block_table` 是 page/block 级映射；HiSparse hot buffer 是 token 级紧凑 resident slots。`hot_slot_0` 可以存 `token_5`，`hot_slot_1` 可以存 `token_300`，它们原本属于不同 logical blocks，但在 hot buffer 中相邻。若强行保留原始 page 结构，就会退回 block residency，放大存储和 copy。

4. **per-layer hot set 不同**
   HiSparse 是 per-layer residency。不同层的 indexer top-k 和 LRU 状态可以不同：

   ```text
   layer 10 hot set = {token_5, token_20, token_300}
   layer 20 hot set = {token_8, token_50, token_1000}
   ```

   传统 PA `block_table` 通常是 per-request 的 KV layout 元数据，不表达“同一个 logical token 在不同层 resident 状态不同”。若为每层维护独立 `block_table`，复杂度已经接近直接维护 `top_k_device_locs`。

5. **Graph 友好性**
   `top_k_device_locs` 的 shape 固定为 `[batch, top_k]`，每步只覆盖值，适合 graph replay。完整 `block_table` 大且稀疏有效，还需要处理 invalid entries、eviction 后清理和动态更新，不如固定 shape 的 top-k 工作集表干净。

| 维度 | 传统 `block_table` | `top_k_device_locs` |
|------|--------------------|---------------------|
| 表达范围 | 完整序列 logical blocks | 当前 attention top-k 工作集 |
| 前提假设 | KV 全量常驻 device | 大部分 KV 在 Host，device 只保留 hot set |
| 更新频率 | 低频，随 token append / block allocation 更新 | 每层每步更新 |
| 粒度 | page/block 级 | token / hot slot 级 |
| 层间关系 | 通常 per-request 共享 | per-layer 独立 |
| Graph 形态 | 大表稀疏更新，invalid entry 复杂 | 固定 `[B, K]` 小表，直接覆盖 |

结论：`block_table` 是为“全量常驻、静态分配”的 PA cache 设计的地址翻译；`top_k_device_locs` 是为“部分常驻、动态换入换出、token 级、per-layer”的 HiSparse hot buffer 设计的工作集定位。两者解决的问题不同。注意，这不是说 GPU attention kernel 没有 `block_table` 参数，而是说 HiSparse 的 sparse/hot 路线不能继续用传统 full-resident PA `block_table` 来表达当前 resident 工作集。

对 NPU 的启示是：当前 SFA 的寻址模型仍然是 `sparse_indices`（logical）+ `block_table`（full-resident PA）。如果要做 HiSparse 等价物，核心问题不是 SFA 能不能 token-level sparse compute，而是 SFA 如何接受一个 **不完整、动态变化、token 级、per-layer** 的 hot-resident 寻址输入。

### 4.3 可行路线

注意：本节先讨论 **attention 如何消费 hot buffer**。Host offload / device compressed backing / Graph-first residency 属于差异 1 的执行模型选择，可以和下面的寻址路线组合。路线 A/B 更适合不改 SFA 的 POC 或中间验证；路线 C 才是与 GPU HiSparse 真实结构最一致的生产方向。

#### 路线 A：让 hot buffer 成为同一 PA 地址空间的一部分（最小改动，不推荐主线）

核心思路：把 hot buffer 当作 `kv_cache` 的一个 PA 子区域，用 block allocator 管理 hot slots，并构造 / 改写 `block_table` 指向这些 hot PA blocks。这样 `npu_sparse_flash_attention` 的 `keyGm + block_table + sparse_indices` 寻址链不需要改。

这里必须明确 `sparse_indices` 与 `block_table` 的坐标系，不能混用：

- **A1: 保留原始 logical token positions**。`sparse_indices` 仍然是原始 token id，例如 `5000`；SFA 会算 `5000 / 128 = 39`，所以 `block_table` 必须覆盖原始 logical block id 空间，至少到 `max(logical_token) / block_size`。只有 resident blocks 的 entry 指向 hot slots，其余 entry 可以是 dummy / invalid，因为 top-k 不会访问它们。优点是无需重写 top-k；缺点是 `block_table` 大且稀疏有效，更新/清理复杂。
- **A2: 重映射到 hot view compact positions**。先把 selected blocks / tokens 编成 hot view 内坐标，例如 hot block slot 7、offset 8 对应 sparse index `7 * 128 + 8`；此时 `block_table` 可以 compact，但 `sparse_indices`、`actual_seq_lengths_kv` 和 rope/KV view 都必须使用 hot view 坐标系。优点是表小；缺点是每步多一层 remap，且更容易和原始 sequence 语义混淆。

```
Indexer 产出 logical top-k tokens
  → Python 层：token positions → 所属 block IDs → 去重 → 需要哪些 blocks
  → Host 发起 H2D copy：miss blocks 从 host → hot buffer (via swap_blocks + h2d_stream)
  → Event 同步
  → A1:
      sparse_indices = original_topk_tokens
      block_table = full_logical_hot_block_table
      actual_seq_lengths_kv = original_seq_len
    或 A2:
      sparse_indices = remapped_hot_view_indices
      block_table = compact_hot_block_table
      actual_seq_lengths_kv = hot_view_valid_len
  → npu_sparse_flash_attention(query=..., key=hot_buffer, value=hot_buffer, ...)
```

优点：
- 复用现有 `npu_sparse_flash_attention` op，不改 kernel
- 工程上最接近现有 PA / SFA 调用链
- 如果稀疏策略本身是 block-level，这条路线容易和 baseline 对齐

缺点：
- 对 token-level top-k 来说，hot buffer 是 block superset，不等价于 CUDA HiSparse 的 token-level residency
- 如果 top-k=2048 tokens 分布在很多 block 中，实际需要 swap-in 的 block 数可能远超 2048/128=16
- 即使用 in-block mask 修正 attention 语义，也无法避免 block 级存储和搬运放大
- 需要验证 `npu_sparse_flash_attention` 是否支持 hot buffer 形式的 block_table / sparse_indices

#### 路线 B：Compact PA view（复用 SFA，但每步 gather top-k）

核心思路：把 top-k token 紧凑地 copy 到一个小 PA_BSND 格式 buffer，伪造一个 compact `block_table` 指向该 buffer，再把 `sparse_indices` 改成 `[0, 1, 2, ..., K-1]`。这样可以复用现有 SFA，但 attention 前要先构造 compact view。

```
Indexer 产出 logical top-k tokens
  → residency runtime 确保这些 token 在 device 上可读
  → D2D / H2D gather：top-k tokens → compact_pa_buffer
  → compact_block_table = identity
  → sparse_indices = [0, 1, 2, ..., K-1]
  → npu_sparse_flash_attention(key=compact_pa_buffer, block_table=compact_block_table, ...)
```

优点：
- 保持 token 级精度，与 CUDA HiSparse 语义一致
- 不需要改 SFA kernel 的 attention 数学和 softmax 逻辑
- 可以作为验证 token-level hot residency 语义的中间方案

缺点：
- 每步需要 gather / copy 全部 top-k token，即使其中大部分已经 hit
- 若 compact buffer 仍按 PA block_size=128 对齐，会有 padding / layout 浪费
- 只把 attention 输入变小，未必保留 HiSparse 的 LRU hot buffer 命中收益

#### 路线 C：扩展 SFA 支持 direct physical hot slot addressing（推荐生产方向）

核心思路：保留 SFA 的 attention 计算主体，但新增一种寻址模式：`sparse_indices` 不再解释为 logical token id，而是解释为 hot buffer 内的 physical slot / physical row offset。kernel 跳过 `logical token -> block_table -> PA block` 翻译，直接从 `hot_buffer_ptr + indices[i] * stride` 读取 K/V。

```
Indexer 产出 logical top-k tokens
  → residency op: hit/miss/LRU/copy
  → 输出 top_k_device_locs[B, K]  # physical hot slots
  → npu_sparse_flash_attention(
      key=hot_buffer,
      value=hot_buffer,
      sparse_indices=top_k_device_locs,
      addressing_mode=PHYSICAL_HOT_SLOT,
      block_table=None 或 ignored)
```

优点：
- 与 GPU HiSparse 的真实结构一致：自定义 residency op + 复用 / 扩展已有 sparse attention kernel 的寻址接口
- token 级 residency/copy/compute 都能做到 O(top_k)，不会退化成 block superset
- 不需要重写完整 attention 数学，主要改 SFA 的 `DataCopyPA` / `GetKeyGmOffset` 这类 load-addressing 分支
- fixed-shape `top_k_device_locs[B, K]` 对 Graph replay 更友好

缺点：
- 需要 CANN / AscendC op ABI 和 kernel 修改
- 需要明确 `key_rope`、MLA compressed KV、newest token、invalid index、per-layer hot buffer 的地址语义
- Host-backed miss copy 是否能进 Graph 仍取决于差异 1；否则需要配合 Graph-first residency op 或 device compressed backing store

### 4.4 Block hot buffer Host-offload POC（路线 A 的验证版）

这一路线不是推荐的最终生产形态，而是为了快速验证：Host KV pool、hot buffer、LRU、H2D swap-in、SFA 消费 hot view 这些状态机是否能跑通。

```
Hot buffer: [num_hot_blocks, 128, num_kv_heads, head_size]  PA_BSND layout
Newest slot: 单独的 1-token buffer（当前 step 新写的 token）

每步 decode:
  1. 写 newest token KV 到 newest slot
  2. Indexer 从全量 indexer cache（常驻 NPU，维度小）选 top-k tokens
  3. Python 层：top-k tokens → 所属 blocks → hit/miss 判定（host 侧）
  4. Miss blocks: H2D copy via h2d_stream + swap_blocks
  5. Event 同步
  6. 构造 hot_block_table + 把 newest slot 追加
  7. npu_sparse_flash_attention(hot_buffer + newest, hot_block_table, ...)
  8. 异步 D2H backup newest token → host pool
```

优点：
- 复用现有 SFA op
- Block 级 swap-in 减少 copy 调用次数，但会增加单次 copy 字节数
- Indexer KV cache 常驻 NPU（维度小，index_head_dim=128，成本低）
- Newest token 特殊处理保证正确性

缺点：
- Block 级粒度意味着 hot buffer 实际容量 > top_k tokens（因为一个 block 里可能只有部分 token 被选中）
- Hit/miss 判定在 host 侧，有 Python 开销（第一版可接受，后续可下沉到 AscendC）
- 不满足 HiSparse 端到端 O(top_k) residency/copy 目标，不应作为 Graph-first 生产主线

#### 4.4.1 数据结构

```
┌─────────────────────────────────────────────────────────────────┐
│  Host (Pinned CPU Memory)                                       │
│                                                                 │
│  HostKVPool:                                                    │
│    shape: [layer_num, max_host_blocks, 128, num_kv_heads, dim]  │
│    存储全量 KV（按 block 组织）                                  │
│                                                                 │
│  host_block_table[req, logical_block_idx] -> host_block_slot    │
└─────────────────────────────────────────────────────────────────┘
                              ▲ D2H backup (异步)
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│  NPU Device Memory          │                                   │
│                                                                 │
│  Hot Buffer (shared across requests):                           │
│    shape: [num_hot_blocks, 128, num_kv_heads, dim]              │
│    PA_BSND layout，可直接传给 npu_sparse_flash_attention        │
│                                                                 │
│  Newest Buffer (per request):                                   │
│    当前 step 新写的 token KV                                    │
│                                                                 │
│  Indexer KV Cache (全量常驻):                                   │
│    shape: [num_blocks, 128, 1, index_head_dim=128]              │
│    维度小，常驻成本低                                           │
│                                                                 │
│  Hot Metadata (host 侧维护):                                   │
│    hot_block_tokens[block_slot] -> set of token positions       │
│    lru_queue: block 级 LRU                                      │
│    req_hot_blocks[req] -> allocated hot block slots              │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.4.2 每步 Decode 流程

```python
def decode_step(forward_batch):
    # 1. 写 newest token KV
    write_kv_to_newest_slot(hidden_states, newest_buffer)
    
    # 2. Indexer 选 top-k（indexer KV 常驻 NPU，不需要 swap）
    topk_tokens = npu_lightning_indexer(q_li, indexer_kv_cache, ...)
    # topk_tokens: [batch, 2048] — logical token indices when sparse_block_size=1
    
    # 3. Hit/miss 判定（host 侧）
    needed_blocks = compute_needed_blocks(topk_tokens)  # 去重
    hit_blocks = needed_blocks & current_hot_blocks
    miss_blocks = needed_blocks - hit_blocks
    
    # 4. LRU 驱逐 + H2D copy
    if len(miss_blocks) > 0:
        evict_blocks = lru_evict(len(miss_blocks))
        with torch.npu.stream(h2d_stream):
            swap_blocks(host_pool, hot_buffer, miss_blocks → evict_slots)
        h2d_event = h2d_stream.record_event()
        torch.npu.current_stream().wait_event(h2d_event)
    
    # 5. 构造 hot view
    # POC 建议用 A2：把 hot blocks 编成 compact PA view，避免传原始 seq_len 尺度的大 block_table。
    hot_block_table = build_compact_block_table(needed_blocks, hot_buffer_mapping)
    hot_valid_token_count = count_valid_tokens_in_hot_view(needed_blocks, seq_len)
    hot_sparse_indices = torch.arange(
        hot_valid_token_count, dtype=torch.int32, device=ql_nope.device
    ).view(1, -1).expand(batch_size, -1)
    
    # 6. SFA attention
    attn_output = npu_sparse_flash_attention(
        query=ql_nope,
        key=hot_buffer,        # 只有 hot blocks
        value=hot_buffer,
        sparse_indices=hot_sparse_indices,  # [0, 1, ..., hot_valid_token_count - 1]
        block_table=hot_block_table,
        actual_seq_lengths_kv=hot_valid_token_count,  # hot view 长度，不是原始 seq_len
        query_rope=q_pe,
        key_rope=hot_buffer_rope,
        layout_kv="PA_BSND")
    
    # 7. 异步 D2H backup newest token
    with torch.npu.stream(d2h_stream):
        backup_newest_to_host(newest_buffer, host_pool)
```

#### 4.4.3 Indexer KV 为什么可以常驻

Indexer KV cache 的显存成本：
```
indexer_kv = num_blocks × 128 × 1 × index_head_dim × dtype_size
           = (max_seq_len/128) × 128 × 1 × 128 × 2 bytes
           = max_seq_len × 256 bytes

128K tokens → 128K × 256 = 32 MB per layer
60 layers → 1.9 GB total
```

对比 attention KV cache：
```
attention_kv = max_seq_len × (kv_lora_rank + qk_rope_head_dim) × 2 bytes
             = 128K × (512 + 64) × 2 = 144 MB per layer
60 layers → 8.4 GB total
```

Indexer KV 只占 attention KV 的 ~22%，且是 top-k 选择的必要输入。第一版让它常驻是合理的。

#### 4.4.4 与现有 NPU DSA 路径的集成点

当前 NPU DSA 路径（`forward_dsa_prepare_npu` + `forward_dsa_core_npu`）的修改点：

1. **KV 写入**：从"写入全量 kv_cache"改为"写入 newest slot + 异步 D2H backup"
2. **Indexer 调用**：不变（indexer KV 常驻，`npu_lightning_indexer` 调用不变）
3. **Attention 调用**：从"传全量 kv_cache + sparse_indices"改为"传 hot_buffer + hot_block_table"
4. **新增**：hit/miss 判定 + H2D swap-in + LRU 管理

### 4.5 关键风险和验证点

| 风险 | 验证方法 |
|------|----------|
| `npu_sparse_flash_attention` 能否对 hot buffer 做 compact-view attention | 构造小 hot buffer，`sparse_indices=[0..hot_valid_len-1]`，`actual_seq_lengths_kv=hot_valid_len`，对比 full-resident 输出 |
| Block 级粒度下 hit rate 是否足够 | 统计 top-k tokens 的 block 分布，计算 block-level hit rate |
| Compact PA view 的 gather 成本是否可接受 | Benchmark top-k gather / layout transform，和 SFA 时间分开计量 |
| Direct physical hot slot 模式能否只改寻址分支 | 在 SFA kernel 中定位 `DataCopyPA` / `GetKeyGmOffset`，做最小 ABI 原型 |
| `key_rope` / newest token / invalid index 语义是否能对齐 | 构造包含 newest、padding、越界 index 的单层 golden test |
| H2D copy 延迟是否可接受 | Benchmark `swap_blocks` 在 h2d_stream 上的延迟 vs decode step 总时间 |
| Newest token 追加到 hot buffer 的语义是否正确 | 验证 SFA op 能否处理"hot buffer + 1 extra token"的 actual_seq_lengths |
| Host 侧 hit/miss 判定的 Python 开销 | Profile，如果成为瓶颈则下沉到 AscendC custom op |

### 4.6 分阶段落地

> 注：以下阶段是 **POC 验证顺序**，不是生产部署路径。P0-P3 使用 Host offload 快速验证端到端正确性；P4 转向 Graph-first residency 验证性能上限。最终生产路径取决于 P4 结论——若 CANN Graph 能力满足，则走 Graph-first；否则可能回退到 Host offload 或转向 Device compressed backing。三个方向的关系见 4.2 节。

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0: SFA 寻址探针** | 验证现有 SFA 能否消费 hot PA view；定位 direct physical slot 最小改动点 | 单层 attention golden test + kernel call graph |
| **P1: Block hot buffer POC** | HostKVPool + HotKVPool + LRU + block-level H2D swap-in | 跑通正确性，测量 block 放大倍数 |
| **P2: Compact PA view 验证** | top-k token gather 到 compact buffer，复用 SFA | token-level 语义正确性 + gather 成本 |
| **P3: Direct physical SFA 原型** | 扩展 SFA 寻址 ABI，消费 `top_k_device_locs` | 单层 / 多层 decode 对齐 full-resident |
| **P4: Graph-first residency** | hit/miss/LRU/copy 下沉到固定 shape op，减少 Host 控制面 | graph replay 可行性 + 分段 benchmark |
| **P5: 生产化** | 多请求、preempt/abort、资源管理、异常回收 | 压力测试 |

### 4.7 验收标准

1. **正确性**：同 prompt、同 seed，HiSparse 输出 vs full-resident 输出的 logits max diff < 1e-3
2. **资源**：NPU KV 显存 ∝ hot resident set（token slots 或 hot blocks）而非 `max_seq_len`
3. **性能**：拆出 indexer / residency update / swap-in or gather / SFA / backup 五段耗时，观察 hit rate、miss copy 和寻址改造开销


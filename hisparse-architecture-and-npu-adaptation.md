# SGLang HiSparse 能力解读、架构分析与 NPU 适配方案

## 概述

### HiSparse 是什么

HiSparse 解决的问题：长序列 decode 时 KV cache 全量常驻 GPU 显存太贵。

HiSparse 的做法：全量 KV 放 CPU pinned memory，GPU 上只保留一个固定大小的 hot buffer（典型 4096 slots）。每步 decode，模型自带的 indexer 选出 top-k 个最重要的 token，HiSparse 确保这些 token 的 KV 在 GPU 上可读（miss 的从 CPU 搬过来），然后 attention 只读 hot buffer。

效果：GPU KV 显存从 O(seq_len) 降到 O(top_k)。128K 上下文、buffer_size=4096 时，省约 30 倍。

核心语义不是 sparse attention（那是 indexer 的事），而是 **sparse KV residency**——控制哪些 KV 在 device 上、哪些不在。

### GPU HiSparse 怎么做的

三步：

```text
1. Indexer 选出 top-k logical token positions
2. Swap-in kernel: hit/miss/LRU/Host→Device copy → 输出 top_k_device_locs（physical hot slots）
3. FlashMLA: indices = top_k_device_locs → 直接按 physical offset 读 KV tensor
```

为什么这么简洁：

- **Swap-in kernel 不是 attention kernel。** 它只做 residency 管理和数据搬运，attention 计算完全复用 FlashMLA / FA3 / tilelang。
- **Hot buffer 是 KV cache tensor 的子区域，不是独立 tensor。** 所以 swap-in 输出的 physical slot 和 FlashMLA 的 `indices` 参数语义天然一致，不需要任何适配。
- **全部在一个 kernel 里完成。** GPU kernel 用 PTX `ld.global.nc` 直接从 pinned host memory 读，不需要 CPU 参与，不打断 CUDA Graph。

### NPU 当前差什么

NPU 的 `npu_sparse_flash_attention` 在 `sparse_block_size=1` 时已经能做 token 级 sparse attention。**NPU 不缺 sparse compute，缺的是 sparse residency。**

当前 NPU 路径：indexer 选出 top-k → SFA 只对这些 token 做 attention，但被选 KV 仍然假设全量常驻 NPU。没有 hot buffer、没有 LRU、没有 swap-in。

寻址模型的差异是根本原因：

| | GPU FlashMLA | NPU SFA |
|---|---|---|
| indices 含义 | physical hot slot offset | logical token id |
| 地址翻译 | 无（直接按 offset 读） | sparse_indices → block_table → PA block → keyGm |
| 隐含前提 | 只读 hot buffer 中的 resident token | 全量 KV 在 device 上，block_table 是完整页表 |

HiSparse 启用后"block_table 是完整页表"这个前提被破坏。128K token 只有 4096 在 device 上，大部分 logical token 在 device 上没有对应的 physical slot。

### NPU 适配要解决两层问题

**第一层：执行模型——residency runtime 怎么跑**

三个方向（并列选择，不是串行阶段）：

| 方向 | 做法 | 适用场景 |
|------|------|----------|
| Graph-first fused op | hit/miss/LRU/copy 下沉到 AscendC op 或 CANN Graph 内 | 性能上限最高，需要验证 CANN 能力 |
| Device compressed backing | 全量 KV 压缩后常驻 NPU，miss 时 D2D copy/decompress | 更 NPU-native，不依赖 Host offload |
| Explicit Host offload | Host 侧发起 H2D copy | 最适合 POC，但打断 Graph |

注意：这三者是岔路口，不是递进关系。选了 device compressed backing 就不需要 Host offload 的整套路径。

**第二层：attention 寻址——SFA 怎么读 hot buffer**

三条路线：

- **路线 A**：把 hot buffer 放进 PA 地址空间，构造/改写 block_table。不改 SFA kernel，但退化成 block 粒度 residency，存储和搬运会放大。
- **路线 B**：每步把 top-k token gather 到一个 compact PA buffer，sparse_indices 改成 [0,1,...,K-1]。token 精度，但每步有全量 gather 开销。
- **路线 C（推荐）**：扩展 SFA 支持 physical hot slot addressing。sparse_indices 不再是 logical token id，而是 hot buffer 内的 physical slot offset。SFA 跳过 block_table 翻译，直接按 offset 读。改动集中在 SFA 的 load/addressing 分支，不动 attention 数学。

路线 C 和 GPU HiSparse 的结构一致：自定义 residency op + 扩展已有 attention kernel 的寻址接口。

### GLM-5 和 DeepSeek-V4 要分开做

SGLang GPU 侧已经是两套独立的 HiSparse 实现。原因不只是"C4 有 4:1 压缩"，而是二者在 attention 架构中的角色根本不同：

- **GLM-5 / V3.2（NSA 路径）**：sparse attention 是该层的**主 attention**。indexer 选出 2048 个原始 token，FlashMLA 只读这些 KV，这就是全部。
- **DeepSeek-V4（DSV4 路径）**：sparse attention 只是复合 attention 的**一个分支**。每层有一个 `compress_ratio`（0、4 或 128），决定该层是 SWA only、SWA + C4 sparse、还是 SWA + C128。HiSparse 只管 `compress_ratio=4` 层的 C4 pool offload，C4 部分作为 `extra_indices_in_kvcache` 传入。

这导致 KV pool 的 token 语义、内存 layout、写入接口、attention 消费方式全不同，不能共用一个 KV pool 实现。

**可以共用的**：Host pool 管理、LRU 状态机、swap-in 调度框架、fixed-shape 输出接口。

**必须分开的**：KV item layout、copy stride 计算、loc 映射语义、attention kernel 参数。

### 落地顺序

**先做 GLM-5 / NSA 通用路径**，因为：NPU 已有 `npu_lightning_indexer` + `npu_sparse_flash_attention` 完整路径，compress_ratio=1 无压缩，layout 简单。

```text
P0: SFA 寻址探针
    验证 SFA 能否消费 hot PA view，定位 direct physical slot 最小改动点

P1: Block hot buffer POC
    跑通 HostKVPool + HotBuffer + LRU + block-level H2D swap-in
    目标是正确性，不是性能

P2: Compact PA view 验证
    top-k gather 到 compact buffer，测 token-level 语义正确性 + gather 成本

P3: Direct physical SFA 原型
    扩展 SFA 寻址 ABI，让 SFA 直接消费 top_k_device_locs

P4: Graph-first residency
    把 hit/miss/LRU/copy 下沉到固定 shape op，脱离 Host 控制面

P5: 生产化
    多请求、preempt/abort、资源管理、压力测试
```

**DeepSeek-V4 不应作为第一条 NPU HiSparse 主线。** 它依赖 DSV4 基础推理（多 pool、C4 indexer、SWA+C4 双路 attention）先在 NPU 上跑通，之后再单独加 C4 HiSparse。

---

## 第一部分：SGLang HiSparse 能力解读

### 1.1 HiSparse 解决什么问题

长序列 decode 时，全量 KV cache 常驻 GPU 显存太贵。以 DeepSeek-V3.2（MLA, kv_lora_rank=512, qk_rope_head_dim=64）为例：

```
单请求 128K tokens 的 KV 显存 ≈ 128K × (512+64) × 2bytes × 60layers ≈ 8.4 GB
```

HiSparse 的核心能力：**只在 GPU 上保留一个固定大小的 "hot buffer"（LRU 缓存），全量 KV 放在 pinned host memory，每步 decode 按 indexer 选出的 top-k token 做 swap-in，GPU KV 显存从 O(seq_len) 降到 O(top_k)。**

### 1.2 资源模型

```
启用 HiSparse:
  GPU KV 显存 = max_num_reqs × (device_buffer_size + 1) × kv_dim × num_layers
  Host KV 内存 = max_num_reqs × max_seq_len × kv_dim × num_layers / compress_ratio

不启用 HiSparse:
  GPU KV 显存 = max_num_reqs × max_seq_len × kv_dim × num_layers
```

当 max_seq_len=128K, device_buffer_size=4096 时，GPU KV 省约 **30x**。

### 1.3 适用范围

- 仅支持 DSA 模型（DeepSeek-V3.2、GLM-5、DeepSeek V4）
- 必须 `--disable-radix-cache`
- 两条路径：NSA 路径（compress_ratio=1）和 DSv4 路径（compress_ratio=4）
- BF16 KV → `flashmla_sparse` backend；FP8 KV → `flashmla_kv` backend

---

## 第二部分：HiSparse 架构分析

### 2.1 数据结构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Host (Pinned CPU Memory, cudaHostRegister 注册, GPU 可直接读)  │
│                                                                 │
│  Host KV Pool:                                                  │
│    shape: [layer_num, max_total_tokens, kv_dim]                 │
│    存储：所有 request 的全量 KV（每个 compressed token 一个 slot）│
│                                                                 │
│  req_to_host_pool[req, compressed_token_pos] -> host_slot       │
└─────────────────────────────────────────────────────────────────┘
                              ▲ D2H backup (每步 newest token)
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│  Device (GPU)               │                                   │
│                                                                 │
│  Hot Buffer (per request, per layer):                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ device_buffer_size 个 slot + 1 个 reserved (newest token)  │ │
│  │ 典型值: top_k=2048, buffer_size=4096                       │ │
│  │                                                            │ │
│  │ 跟踪表：                                                   │ │
│  │   req_device_buffer_tokens[layer, req, slot] -> token_pos  │ │
│  │   req_device_buffer_token_locs[layer, req, slot] -> loc    │ │
│  │   lru_slots[layer, req, :] -> LRU 排序                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Indexer KV Cache (全量常驻 GPU，维度小):                       │
│    FP8 quantized, head_dim=128, 用于 top-k 选择                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 请求生命周期

```
┌─────────────┐     ┌──────────┐     ┌──────────────────────┐     ┌────────┐
│   Prefill   │────>│ Staging  │────>│   Decode (per step)  │────>│ Finish │
└─────────────┘     └──────────┘     └──────────────────────┘     └────────┘
```

| 阶段 | 动作 |
|------|------|
| **Prefill** | KV 写入 device pool（全量常驻 GPU）。Prefill 结束后触发异步 D2H 全量备份到 host pool（`write_staging_stream`） |
| **Staging** | 等待 D2H 完成（poll `finish_event`）。完成后 `alloc_device_buffer`：回收大 pool 的 device 槽位，分配小的 hot buffer（`device_buffer_size` 个 slot）。此时 GPU 显存释放 |
| **Decode** | 每步：1) 备份上一步 newest token 到 host；2) 映射本步新 token 到 reserved slot；3) 每层 attention：indexer 选 top-k → swap-in → sparse attention |
| **Finish** | 释放 host slots + device buffer slots，清理所有 per-req 状态 |

### 2.3 Decode 每步详细流程

```
prepare_for_decode():
  │
  ├── seq_lens += 1
  ├── alloc_decode() → new out_cache_loc (logical)
  │
  └── map_last_loc_to_buffer():
        ├── _eager_backup_previous_token():
        │     如果 seq_len > buffer_size 且到达 compress_ratio 边界：
        │       在 decode_backup_stream 上：
        │         wait(main_stream)
        │         D2H copy previous token → host pool
        │         record _backup_done_event
        │
        └── 映射 newest token 的 out_cache_loc → reserved buffer slot
            更新 full_to_hisparse_device_index_mapping

forward_decode() (per layer):
  │
  ├── 写 KV cache（newest token 写入 reserved slot）
  │
  ├── Indexer 计算 top-k:
  │     q_fp8 = quant(wq_b(q_lora))
  │     logits = fp8_paged_mqa_logits(q_fp8, indexer_kv_cache, weights)
  │     topk_indices = fast_topk(logits, index_topk)
  │     输出: [batch, top_k] 的 logical token positions
  │
  ├── swap_in_selected_pages(topk_indices, layer_id):
  │     CUDA kernel (hisparse.cuh) 内完成：
  │       hit/miss 判定 → LRU 更新 → host→device copy → 输出 device locs
  │     输出: top_k_device_locs [batch, top_k]
  │
  └── Sparse Attention (FlashMLA / tilelang):
        page_table = top_k_device_locs
        对 hot buffer 中的 top_k 个 token 做 attention
```

### 2.4 swap-in 机制（核心 CUDA kernel）

先澄清一个容易误解的点：HiSparse 自定义的 CUDA kernel 主要是 **residency / swap-in kernel**，不是重新实现 attention 数学。attention 计算复用已有的 FlashMLA sparse / FlashMLA KV / tilelang / FA3 等 backend；HiSparse 做的是把 indexer 产生的 logical top-k token 转换成 hot buffer 中的 physical device locations，再交给这些已有 attention kernel 的 `indices` / `page_table` 类接口。

这是 HiSparse 最关键的设计——**一个 CUDA kernel 内原子完成 hit/miss 判定 + LRU 更新 + host→device copy**：

```
Kernel: load_cache_to_device_buffer_kernel
  1 CUDA block = 1 request

输入：
  top_k_tokens[num_reqs, top_k]       indexer 选出的 token positions
  device_buffer_tokens[req, slot]      当前 hot buffer 里缓存了哪些 token
  host_cache[host_slot]                pinned host memory (GPU 可直接读)
  lru_slots[req, :]                    LRU 排序

算法：
  1. Fast path: 如果 seq_len <= buffer_size，所有 token 都在 buffer，直接查表
  2. 构建 shared memory hash table（top-k tokens, Knuth multiplicative hash）
  3. 扫描 LRU slots：
     - 命中 hash table → HIT，记录 device loc，标记 MRU
     - 未命中 → 加入 evictable 列表（LIFO）
  4. 扫描 top-k tokens：
     - 未标记 HIT → MISS，配对一个 evictable slot
  5. 重写 LRU 顺序：stale slots → miss slots → hit slots
  6. 每个 warp 处理一个 MISS：
     用 PTX ld.global.nc.v2.b64 / st.global.cg.v2.b64
     从 pinned host memory 拷贝到 device buffer（128-bit vectorized）

输出：top_k_device_locs[num_reqs, top_k]  (physical device buffer locations)
```

关键特性：
- **GPU-initiated DMA**：kernel 内部直接从 pinned memory 读，不需要 CPU 参与
- **固定 shape**：适合 CUDA graph capture，无 Python 分支
- **per-layer 独立**：每层有自己的 LRU 状态，不同层可以缓存不同 token
- **粒度是 token 级**：每个 sparse index 指向一个 token（page_size=1）
- **newest token 特殊处理**：始终在 reserved slot（buffer_size 位置），不参与 LRU

#### 2.4.1 源码实现细节（经逐项验证）

**PTX 指令**

swap-in kernel 的 miss copy 使用手写 PTX inline assembly（`hisparse.cuh` line 37-51）：

```c
// 从 Host pinned memory 读（non-coherent，走 texture cache 做 streaming read）
asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(s) : "memory");

// 写入 GPU hot buffer（cache-global evict-first，写穿 L2 不污染 L1）
asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(d), "l"(lo), "l"(hi) : "memory");
```

- `.nc`（non-coherent）：利用 GPU texture cache 做 streaming read，适合从 PCIe BAR 映射的 Host 地址读大块连续数据
- `.cg`（cache-global）：evict-first store，写穿 L2 避免 L1 cache 被 miss copy 数据污染
- `.v2.b64`：每条指令搬 128 bit（16 bytes），warp 32 个 lane 并行 = 每轮 512 bytes

copy 工作由 `transfer_item_warp` 函数分配给 warp 的各 lane，stride-by-WARP_SIZE 循环处理一个 item 的全部字节。miss 总量由 kernel 内部计算后分配给各 warp（`hisparse.cuh` line 372-398）。

**Host memory 管理**

Host KV pool 使用 `cudaHostRegister` 将 CPU 内存 pin 住并注册（`memory_pool_host.py` 的 `alloc_with_host_register`）：

```python
torch.cuda.cudart().cudaHostRegister(
    buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
)
```

注册后的内存通过 UVA（Unified Virtual Addressing）暴露给 GPU，kernel 可以直接用虚拟地址访问。不需要 Host 侧 `cudaMemcpyAsync` 参与 swap-in。

**零 Python 逻辑**

`swap_in_selected_pages()`（`hisparse_coordinator.py` line 765-803）只做两件事：
1. 用 -1 填充输出 buffer
2. 调用 JIT kernel

没有 hit/miss 判定、LRU 维护、miss plan 生成等 Python 代码。所有 residency 逻辑都在 kernel shared memory 里完成：
- shared memory hash table 建 top-k token 集合（Knuth multiplicative hash）
- 扫描 device buffer slots 判定 hit/miss
- LIFO 方式收集 evictable slots
- 分配 miss → eviction slot 配对
- 更新 LRU 顺序

**Fixed-shape 与 Graph 兼容性**

kernel 模板参数（`BLOCK_SIZE`、`NUM_TOP_K`、`HOT_BUFFER_SIZE`、`IsMLA`、`IsDsv4Layout`）全部是编译期常量。运行时分支只有两类：
- `seq_len <= HOT_BUFFER_SIZE`：所有 token 都在 buffer 的 fast path（GPU 侧判定，不改 kernel signature）
- `bid >= num_real_reqs[0]`：padded request 跳过（支持 batch padding for Graph）

两者都是 GPU 侧数据驱动的 branch，不需要 Host 参与决策，不破坏 CUDA Graph capture/replay。

**D2H backup 是独立路径**

swap-in（Host→Device）在 kernel 内完成。但 backup（Device→Host）是 Host 侧发起的 copy，使用独立的 `decode_backup_stream`（`hisparse_coordinator.py` 的 `_eager_backup_previous_token`）。两个方向机制不同：
- swap-in：GPU kernel 内 `ld.global.nc` 从 Host 读 → 无 Host 参与
- backup：Host 发起 `backup_from_device_all_layer()` D2H copy → 需要 stream 同步

**FlashMLA 为什么不需要修改**

关键实现细节：**hot buffer 是 KV cache tensor 的一个子区域**，不是独立的 tensor。

`device_buffer_locs` 是 KV cache pool 内的 slot indices。swap-in kernel 把 Host 数据写到 `device_buffer[slot]`，输出的 `top_k_device_locs` 就是这些 slot 在 flattened KV cache tensor 内的 physical offset。FlashMLA 的 `indices` 参数语义正是"KV cache tensor 内的 physical row offset"：

```python
# FlashMLA 测试中 indices 的构造方式（test_flashmla.py line 447-468）：
cur_blocked_indices = block_table_cpu[i, cur_abs_indices // block_size] * block_size + (cur_abs_indices % block_size)
indices_in_kvcache[i, j, :] = cur_blocked_indices
```

FlashMLA `indices` 的 shape 是 `[batch, seq_len_q, topk]`（decode 时 `seq_len_q=1`）。HiSparse 的 `top_k_device_locs` shape 是 `[batch, topk]`，只需 `unsqueeze(1)` 即可对齐。kernel 拿到 physical offset 后直接按地址读 KV tensor，不做任何额外翻译。

因此 HiSparse 不需要改 FlashMLA——因为 swap-in kernel 输出的 physical locations 和 FlashMLA 原本期望的 `indices` 语义完全一致。这也是 GPU 方案能做到"只写一个 swap-in kernel 就完成整个 HiSparse"的根本原因。

**DSv4 路径的 page_size 差异**

NSA 路径（DeepSeek V3.2）的 hot buffer 管理粒度是 1 token（`page_size=1`）。DSv4 路径因为 `compress_ratio=4`，管理粒度是 1 compressed item（代表 4 个原始 token）。严格来说 HiSparse 的 residency 粒度是 O(top_k compressed items) 而非 O(top_k raw tokens)，但量级不变。

### 2.5 Stream / 同步模型

```
  Main Stream (forward)          write_staging_stream         decode_backup_stream
  ─────────────────────          ────────────────────         ────────────────────
  Prefill forward                                             
       │                                                      
       ├─ start_event.record()                                
       │                         wait(start_event)            
       │                         D2H all-layer backup         
       │                         finish_event.record()        
       │                                                      
  ... (other batches) ...                                     
       │                                                      
  collect_ready_reqs()                                        
  (polls finish_event.query())                                
       │                                                      
  Decode forward                                              
       │                                                      
  map_last_loc_to_buffer()                                    
       │                                                wait(main_stream)
       │                                                wait(decode_producer)
       │                                                D2H backup prev token
       │                                                _backup_done_event.record()
       │                                                      
  wait_for_pending_backup()                                   
  (main stream waits _backup_done_event)                      
       │                                                      
  forward_decode()                                            
  (swap_in_selected_pages per layer — on main stream)         
```

### 2.6 Indexer 与 HiSparse 的关系

Indexer 是 DSA 模型架构的一部分（模型训练时就有），HiSparse 是推理优化。两者关系：

- **Indexer**：每层计算 attention score，选出 top-k 个最重要的 token positions。这是模型本身的 sparse attention 机制，不管有没有 HiSparse 都会运行。
- **HiSparse**：利用 indexer 的 top-k 输出，只把这些 token 的 KV 保留在 GPU 上。

不启用 HiSparse 时：indexer 选出 top-k → page table 翻译 → attention 从全量 KV cache 中读取 top-k 位置。
启用 HiSparse 时：indexer 选出 top-k → swap-in kernel 把 miss 的从 host 拉到 hot buffer → attention 从 hot buffer 读取。

关键区别：
- 不启用时，`topk_indices` 经过 `transform_index_page_table_decode` 翻译成 physical locs
- 启用时，强制 `force_unfused_topk=True`（indexer 返回 raw logical positions），然后由 swap-in kernel 返回 physical device buffer locs

### 2.7 两条路径对比

| | NSA 路径 (DeepSeek V3.2, GLM-5) | DSv4 路径 (DeepSeek V4) |
|---|---|---|
| compress_ratio | 1 | 4 |
| index_topk | 2048 tokens | 512 compressed tokens (= 2048 原始 tokens) |
| 备份频率 | 每步 | 每 4 步（compress_ratio 边界） |
| Device pool | HiSparseNSATokenToKVPool | HiSparseC4DevicePool |
| Host pool | MLATokenToKVPoolHost | DeepSeekV4SingleKVPoolHost |
| Swap-in kernel | `load_cache_to_device_buffer_mla` | `load_cache_to_device_buffer_dsv4_mla` |
| D2H kernel | `transfer_kv_all_layer_mla` (sgl_kernel) | `hisparse_offload_to_host` (jit_kernel) |
| Attention backend | FlashMLA sparse / tilelang | `flash_mla.flash_mla_with_kvcache`（SWA indices + extra_indices 双路） |
| `_grow_device_buffers` | 有（短序列动态增长） | 无（固定 padded_buffer_size） |

### 2.8 配置参数

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2 \
  --enable-hisparse \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}' \
  --disable-radix-cache
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `top_k` | 2048（或 model config 的 `index_topk`） | 每步每层选多少 token |
| `device_buffer_size` | `2 × top_k` | hot buffer 大小，必须 >= top_k |
| `host_to_device_ratio` | 2 | 逻辑池大小 = device pool × ratio，决定最大序列长度 |

### 2.9 关键源文件

| 文件 | 职责 |
|------|------|
| `srt/managers/hisparse_coordinator.py` | 中心协调器：生命周期、备份、swap-in 编排 |
| `srt/mem_cache/hisparse_memory_pool.py` | Device pool、allocator、mapping |
| `srt/mem_cache/deepseek_v4_memory_pool.py` | DSv4 专用 pool 和 allocator |
| `srt/mem_cache/memory_pool_host.py` | Host pool（MLATokenToKVPoolHost） |
| `jit_kernel/csrc/hisparse.cuh` | swap-in CUDA kernel |
| `srt/layers/attention/nsa/nsa_indexer.py` | NSA Indexer（top-k 选择） |
| `srt/layers/attention/nsa_backend.py` | NSA attention backend（decode 路径集成） |
| `srt/layers/attention/dsv4/indexer.py` | DSv4 C4 Indexer |

---

## 第三部分：SGLang Ascend NPU 当前 DSA 实现

### 3.1 SGLang 已有 NPU DSA 路径

SGLang 已经有一条完整的 NPU DSA 推理路径（不含 HiSparse），位于 `hardware_backend/npu/`：

```
forward_dsa_prepare_npu():
  │
  ├── fused_qkv_a_proj → q_c + latent_cache
  ├── q_a_layernorm(q) → q_lora
  ├── q_b_proj(q_lora) → q [num_heads, qk_head_dim]
  ├── split → q_nope, q_pe
  ├── BMM(q_nope, w_kc) → q_nope_out (absorbed query)
  ├── rotary_emb → q_pe, k_pe
  │
  └── m.indexer(hidden_states, q_lora, positions, forward_batch, layer_id)
        → topk_indices
        内部调用 torch_npu.npu_lightning_indexer()

forward_dsa_core_npu():
  │
  ├── m.attn_mqa(q_nope_out, k_nope, k_nope, forward_batch,
  │              save_kv_cache=True, q_rope=q_pe, k_rope=k_pe,
  │              topk_indices=topk_indices)
  │     内部调用 torch_npu.npu_sparse_flash_attention()
  │
  ├── BMM(attn_output, w_vc) → V up-projection
  └── o_proj → output
```

### 3.2 NPU DSA 与 CUDA DSA 的关键差异

| 维度 | CUDA DSA (NSA backend) | NPU DSA (AscendAttnBackend) |
|------|----------------------|---------------------------|
| Indexer | `Indexer.forward_cuda()`: FP8 paged MQA logits + fast_topk | `Indexer.forward_npu()`: `npu_lightning_indexer` |
| Attention | FlashMLA sparse / tilelang / FA3 | `npu_sparse_flash_attention` |
| KV cache layout | flat token pool, page_size=1 | PA_BSND, block_size=128 |
| sparse 粒度 | token 级（1 index = 1 token） | 由 `sparse_block_size` 决定；当前 SFA 调用 `sparse_block_size=1`，也是 token 级 |
| topk 含义 | 2048 个 token | 2048 个 logical KV spans；当 `sparse_block_size=1` 时是 2048 个 token |
| HiSparse 集成 | 有（swap-in kernel） | **无** |
| KV 常驻 | 启用 HiSparse 时不常驻 | 全量常驻 NPU |

### 3.3 NPU 路径的核心限制

1. **全量 KV 常驻 NPU**：当前 NPU DSA 只省计算量（sparse attention），不省显存
2. **sparse attention 不等于 sparse residency**：`npu_sparse_flash_attention` 在 `sparse_block_size=1` 时可以 token 级选择 KV，但被选 KV 仍来自全量常驻 NPU 的 PA cache
3. **无 GPU-initiated DMA**：Ascend NPU 不支持 device 端直接读 host memory，H2D copy 必须由 host 发起
4. **PA block_size 与 SFA sparse_block_size 是两层概念**：PA block_size=128 是 KV cache 物理分页粒度，`sparse_block_size=1` 是 sparse attention 的逻辑选择粒度

---

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
- FlashMLA wrapper / backend 本身已有 `indices` 参数，shape 是 `[batch, seq_len_q, topk]`；decode 时 `seq_len_q=1`，因此 HiSparse 的 `top_k_device_locs[B, K]` 只需 `unsqueeze(1)` 就能对齐接口。
- FlashMLA 测试里 `indices_in_kvcache = block_table[logical_block] * block_size + offset`，也就是 KV cache tensor 内的 physical slot / flattened location，而不是 logical token id。HiSparse 不需要改 FlashMLA kernel，原因正是 swap-in kernel 输出的 `top_k_device_locs` 已经是 hot buffer / KV cache tensor 内可直接读取的 physical locations。
- **关键实现细节：hot buffer 就是 KV cache tensor 的子区域，不是独立 tensor。** `device_buffer_locs` 是 KV cache pool 中分配出来的 slot indices；swap-in kernel 把 Host 数据写入 `kv_pool[device_buffer_locs[slot]]`，输出的 `top_k_device_locs` 是这些 slot 在 flattened KV cache tensor 内的 physical offset。FlashMLA 的 `indices` 参数指向同一个 KV cache tensor，语义天然兼容——这是 GPU 方案能做到"只写一个 swap-in kernel 就完成整个 HiSparse"的根本原因。NPU 方案若想走类似路线，也需要让 hot buffer 与 SFA 的 `key`/`value` GM 指针指向同一块地址空间（或扩展 SFA 支持额外地址源）。
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

---

## 第五部分：NSA 模型与 DeepSeek-V4 的 HiSparse 路径差异

### 5.1 结论

GLM-5（及 DeepSeek-V3.2 等 NSA 模型）和 DeepSeek-V4 在 SGLang GPU 侧已经是两套独立的 HiSparse 实现。NPU 适配时也必须分开。

**公共层可以共享**：Host KV pool 管理、device hot buffer 分配、hit/miss/LRU 逻辑、固定 shape 输出 `top_k_device_locs`、swap-in 调度框架。

**模型接入层必须分开**：KV 压缩比、swap-in kernel layout、attention backend 接口、KV pool 结构完全不同。

### 5.2 SGLang GPU 侧两条 HiSparse 路径对比

源码证据：

- 模型分类：`model_config.py` 的 `is_deepseek_nsa()` 包含 `GlmMoeDsaForCausalLM`、`DeepseekV3ForCausalLM`、`DeepseekV32ForCausalLM` 等；`is_deepseek_v4()` 只包含 `DeepseekV4ForCausalLM`。
- 验证分叉：`hisparse_hook.py:60-72`，V4 直接 early return 跳过 NSA 的 dtype 检查。
- 协调器分叉：`hisparse_coordinator.py` 通过 `self.is_dsv4_hisparse` 区分两条路径。

| 维度 | NSA 路径（GLM-5, DeepSeek V3.2） | DSV4 路径（DeepSeek V4） |
|------|----------------------------------|--------------------------|
| 模型类 | `GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM` | `DeepseekV4ForCausalLM` |
| compress_ratio | 1（无压缩，1 token = 1 KV entry） | 4（4 token 压缩为 1 KV entry） |
| Device pool | `HiSparseNSATokenToKVPool` | `HiSparseC4DevicePool`（仅 C4 pool 被替换） |
| Host pool | `MLATokenToKVPoolHost`（layout="layer_first"） | `DeepSeekV4SingleKVPoolHost` |
| swap-in kernel | `load_cache_to_device_buffer_mla`（device+host 都是 linear layout） | `load_cache_to_device_buffer_dsv4_mla`（page-padded device + linear host） |
| Indexer | `nsa_indexer.py`：FP8 paged MQA logits → `fast_topk_v2` | `dsv4/indexer.py`：`topk_transform_512` → raw indices |
| force_unfused_topk | HiSparse decode 时强制 True（返回 raw token positions） | 同上 |
| Attention backend | `nsa_backend.py` → FlashMLA sparse / FA3 / tilelang | `deepseek_v4_backend.py` → FlashMLA with `extra_indices_in_kvcache` |
| FlashMLA 接口 | `page_table_1 = top_k_device_locs`（直接作为 `indices`） | `extra_indices_in_kvcache = top_k_device_locs`（作为额外稀疏部分） |
| KV offload 范围 | 全部 MLA KV | 仅 C4 KV（SWA/C128/indexer pool 不 offload） |
| item_size_bytes | `token_stride_size`（per-token byte footprint） | `kv_cache_total_dim * dtype.itemsize` |

### 5.3 NSA 路径的 HiSparse 端到端流程（GLM-5 / V3.2）

```
Prefill:
  KV 写入 device pool（全量常驻）
  → admit_request_into_staging()
  → 异步 D2H 全量备份到 host pool
  → 分配 device buffer（小 hot buffer）

Decode (per layer):
  nsa_indexer.forward_decode():
    FP8 paged MQA logits（against full index k_cache）
    → fast_topk_v2（force_unfused=True）
    → raw token positions [num_reqs, index_topk]

  nsa_backend.forward_decode() (line 1612):
    page_table_1 = hisparse_coordinator.swap_in_selected_pages(
        req_pool_indices, seq_lens, topk_indices, layer_id)

  swap_in_selected_pages():
    → load_cache_to_device_buffer_mla (CUDA kernel)
    → hit/miss/LRU/copy in shared memory
    → returns top_k_device_locs [num_reqs, top_k]

  Sparse Attention:
    page_table_1 = top_k_device_locs
    → flash_mla_sparse_fwd / flash_attn_with_kvcache / tilelang
    → page_size=1, 每个 entry 是 KV buffer 内的 physical index

  Backup:
    _eager_backup_previous_token()
    → D2H copy newest token to host pool
```

### 5.4 DSV4 路径的 HiSparse 端到端流程

```
Prefill:
  KV 写入 DeepSeekV4TokenToKVPool（含 swa/c4/c128/indexer 四个 sub-pool）
  → C4 pool 的 KV 异步备份到 DeepSeekV4SingleKVPoolHost
  → C4 pool 替换为 HiSparseC4DevicePool（小 hot buffer）

Decode (per layer):
  dsv4/indexer.py:
    topk_transform_512(logits) → raw_indices
    raw_indices 存入 hisparse_coordinator.raw_indices_buffer

  dsv4/indexer.py (line 446-458):
    core_metadata.c4_sparse_page_indices = hisparse_coordinator.swap_in_selected_pages(
        req_pool_indices, c4_seq_lens, raw_indices, compress_layer_id)

  swap_in_selected_pages():
    → load_cache_to_device_buffer_dsv4_mla (CUDA kernel, page-padded layout)
    → returns top_k_device_locs

  deepseek_v4_backend.py (line 1036):
    extra_indices = core_attn_metadata.c4_sparse_page_indices
    o = flash_mla.flash_mla_with_kvcache(
        indices=swa_page_indices,              # SWA 部分（不走 HiSparse）
        extra_indices_in_kvcache=extra_indices, # C4 HiSparse 部分
        extra_topk_length=extra_topk_lengths)
```

注意：DSV4 的 FlashMLA 调用比 NSA 多一层结构——SWA 部分用 `indices`（sliding window，不 offload），C4 sparse 部分用 `extra_indices_in_kvcache`（HiSparse 管理的 hot buffer）。

### 5.5 NPU 适配的分模型策略

**NSA 模型（GLM-5、DeepSeek-V3.2）优先适配：**

理由：
- NPU 侧已有完整的 NSA/SFA 推理路径（`npu_lightning_indexer` + `npu_sparse_flash_attention`）
- compress_ratio=1，KV 管理简单，无压缩/解压逻辑
- swap-in layout 是 linear 的，实现更直接
- 模型已经在 NPU 上跑通了 inference

适配路径：
```
SGLang NSA HiSparse (GPU 参考实现)
  → 移植 residency 语义到 NPU
  → attention 对接 npu_sparse_flash_attention（通过路线 A/B/C）
```

**DeepSeek-V4 作为未来方向：**

理由：
- 需要在 NPU 上先完成 DSV4 模型的基础推理接入（C4/C128/SWA 多 pool 结构）
- compress_ratio=4 引入压缩算术（`(last_locs - 3) // 4`）
- swap-in 需要 page-padded layout 支持
- FlashMLA 的 `extra_indices_in_kvcache` 接口在 NPU SFA 中没有等价物
- SWA + C4 sparse 双路 attention 的 NPU 实现是独立的大工作量

适配路径：
```
先完成 DSV4 基础推理（多 pool、C4 indexer、SWA+sparse 双路 attention）
  → 再加 HiSparse（仅 C4 pool offload + hot buffer）
  → attention 可能需要新的 NPU kernel（或 SFA 的 DSV4 变体）
```

### 5.6 公共组件复用

尽管接入层不同，以下组件可以设计为模型无关的公共层：

| 组件 | 接口 | NSA 使用 | DSV4 使用 |
|------|------|----------|-----------|
| Host KV pool 管理 | `alloc/free host slots` | `MLATokenToKVPoolHost` 等价 | `DeepSeekV4SingleKVPoolHost` 等价 |
| Device hot buffer 分配 | `alloc/free device buffer slots` | `HiSparseNSATokenToKVPool` 等价 | `HiSparseC4DevicePool` 等价 |
| LRU 状态机 | `hit/miss/evict/update` | per-layer, per-req | per-compress-layer, per-req |
| swap-in 调度 | `swap_in_selected_pages(top_k_result, layer_id) → device_locs` | linear layout | page-padded layout |
| Backup (D2H) | `backup_newest_token(layer_id)` | 每步 1 token | 每 compress_ratio 步 1 compressed item |
| 生命周期管理 | `staging → decode → finish` | compress_ratio=1 | compress_ratio=4 |

公共抽象可以是一个 `HiSparseResidencyRuntime` 基类，参数化 `compress_ratio`、`item_size_bytes`、`layout_type`，由 NSA/DSV4 子类分别实现 layout-specific 的 copy 和 attention 对接。

### 5.7 GLM-5 在 SGLang 中的特化点

GLM-5（`GlmMoeDsaForCausalLM`）直接继承 `DeepseekV2ForCausalLM`，几乎没有 attention 相关的模型特化。在 NSA HiSparse 路径里，唯一的 GLM-5 特化是：

- **Head padding**：`nsa_backend.py:364`——GLM-5 的 `num_q_heads` 可能小于 16（例如 64 heads / TP8 = 8），需要 pad to 16 才能对齐 FlashMLA tile 要求
- **Blackwell workaround**：`server_args.py:1796`——GLM-5 在 Blackwell GPU 上禁用 MHA_ONE_SHOT

除此之外，GLM-5 和 DeepSeek-V3.2 在 HiSparse 路径里完全一致。**NPU HiSparse 方案不需要为 GLM-5 单独设计**，只要做好通用 NSA/SFA 路径的适配即可覆盖所有 `is_deepseek_nsa` 模型。

### 5.8 为什么不能共用同一个 KV pool 实现

结论：**可以共用 HiSparse residency runtime（LRU / hit-miss / host pool / hot buffer / swap-in 调度），不能共用同一个 KV pool 实现**。GLM/NSA 和 DSV4 C4 的 KV pool 绑定了 4 个契约，每一个都不同：

#### 5.8.1 token 语义不同

GLM/NSA 的 HiSparse pool：一个原始 token → 一个 KV slot。`topk_indices` 选的就是原始序列 token id。

源码证据：`HiSparseNSATokenToKVPool.translate_loc_from_full_to_compressed()` 直接 `return full_indices`（`hisparse_memory_pool.py:90`），说明没有压缩 token 空间。

DSV4 C4 pool：4 个原始 token → 1 个 C4 compressed KV。只有 `(full_index + 1) % 4 == 0` 的位置才产生 C4 KV。

源码证据：`HiSparseC4DevicePool.translate_loc_from_full_to_compressed()`（`deepseek_v4_memory_pool.py:199-202`）：

```python
mask = (full_indices + 1) % self.compress_ratio == 0
compressed_indices = full_indices[mask] // self.compress_ratio
```

同一个 `loc=100`，在 GLM pool 里是"第 100 个原始 token"，在 DSV4 C4 pool 里要先判断是否合法再映射到 compressed loc。

#### 5.8.2 内存 layout 不同

GLM/NSA pool 是普通 MLA layout：

```python
bytes_per_token = kv_cache_dim * dtype.itemsize  # hisparse_memory_pool.py:71
```

DSV4 C4 是 packed byte layout（`deepseek_v4_memory_pool.py:91-110`）：

```text
qk_nope FP8 (448) + qk_rope BF16 (64×2) + scale (8) = 584 bytes/token
store_dtype = torch.uint8
device 侧还有 page-padded layout（bytes_per_page_padded = ceil_div(page_size * 584, 576) * 576）
```

GPU HiSparse swap-in kernel 专门用 `IsDsv4Layout` 模板参数区分两种 layout（`hisparse.cuh:90`），因为 device 侧 page-padded 和 host 侧 linear 的 stride 计算不同。

#### 5.8.3 写入接口不同

GLM/NSA pool 写入的是标准 MLA 的 `cache_k_nope + cache_k_rope`：

```python
def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
    loc = self.translate_loc_to_hisparse_device(loc)
    super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
# hisparse_memory_pool.py:102-110
```

DSV4 C4 pool 写入的是已经 pack 好的 `NopeFp8RopeBf16Pack`：

```python
def set_key_buffer(self, layer_id, loc, cache_nope_fp8_rope_bf16_pack):
    loc = self.translate_loc_to_hisparse_device(loc)
    super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)
# deepseek_v4_memory_pool.py:217-224
```

DSV4 C4 pool 不支持 `get_cpu_copy` / `load_cpu_copy`（line 235-238），说明它的 offload/restore 走专用通道而非通用 KV 接口。

#### 5.8.4 attention 消费方式不同

GLM/NSA：`top_k_device_locs` 直接作为 FlashMLA sparse 的 `indices`：

```python
indices_input = page_table_1.unsqueeze(1)  # shape [s_q, h_kv=1, topk]
o, _, _ = flash_mla_sparse_fwd(q=q_input, kv=kv_cache, indices=indices_input, ...)
# nsa_backend.py:1766-1771
```

DSV4 C4：`top_k_device_locs` 作为 FlashMLA 的 `extra_indices_in_kvcache`（额外稀疏分支）：

```python
extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
extra_indices = core_attn_metadata.c4_sparse_page_indices  # HiSparse 输出
o = flash_mla.flash_mla_with_kvcache(
    indices=swa_page_indices,                # SWA 部分（不走 HiSparse）
    extra_indices_in_kvcache=extra_indices,   # C4 HiSparse 部分
    extra_topk_length=extra_topk_lengths)
# deepseek_v4_backend.py:971-975, 1036
```

**GLM 的 sparse indices 是主 KV indices；C4 的 sparse indices 是 extra KV indices。** 这是结构性差异：GLM HiSparse 管的是整个 attention 的 KV 来源；DSV4 C4 HiSparse 只管复合 attention 中 C4 分支的 KV 来源，SWA 和 C128 不受 HiSparse 影响。

#### 5.8.5 如果强行共用的后果

如果试图让 GLM 和 DSV4 C4 共用一个 KV pool 实现，最容易出错的四个点：

1. **`loc` 语义错**：同一个 loc 值在两种 pool 里含义不同
2. **copy stride 错**：584 bytes packed uint8 vs `kv_cache_dim * dtype.itemsize` 的步长计算完全不同
3. **packed layout 被当普通 tensor 读**：如果把 DSV4 的 uint8 packed data 当 BF16/FP8 tensor 解释，读出的 K/V 值无意义
4. **attention kernel 收到错误格式的 KV**：FlashMLA sparse 期望的 `kv_cache` shape 和 DSV4 backend 期望的 `extra_k_cache` shape 完全不同

### 5.9 GLM Sparse Attention 与 DSV4 C4 的结构性差异（超越压缩比）

5.8 节说明了 KV pool 层面的 4 个契约差异。本节从 attention 子系统角度补充 6 个结构性差异，说明二者不只是"C4 有压缩、GLM 没有"，而是在整个 attention 架构中扮演的角色、indexer 语义、HiSparse 接入位置都不同。

#### 5.9.1 不是同一个 attention 子系统

GLM/DSA 走 SGLang 的 NSA 路线。`GlmMoeDsaForCausalLM` 直接继承 `DeepseekV2ForCausalLM`（`glm4_moe.py:1483`），使用 `nsa_backend.py` 的 FlashMLA sparse / FA3 / tilelang。

DSV4 走独立的 `deepseek_v4_backend.py`。每层由 `config.compress_ratios[layer_id]` 决定 attention 类型：`0`（SWA only）、`4`（SWA + C4 sparse）、`128`（SWA + C128）（`deepseek_v4.py:186-192`）。

只有 `compress_ratio == 4` 的层才有 C4Indexer。

#### 5.9.2 Sparse attention 的角色不同

**GLM：sparse attention 是该层的主 attention。** Indexer 选出 top-k token 后，FlashMLA 只读这些 token 的 KV，这就是这一层的完整 attention 输出。

**DSV4 C4：sparse attention 只是复合 attention 的一个分支。** DSV4 每层有一个 `compress_ratio`（由 `config.compress_ratios[layer_id]` 指定），决定该层的 attention 组成：

```text
compress_ratio=0:   SWA only（滑动窗口，无 extra sparse）
compress_ratio=4:   SWA + C4 extra compressed sparse attention (top-512 compressed tokens)
compress_ratio=128: SWA + C128 compressed/global branch
```

三者是逐层互斥的（`deepseek_v4.py:191` `assert compress_ratio in [0, 4, 128]`；`deepseek_v4_backend.py:972-979` if/elif 分支）。HiSparse 只涉及 `compress_ratio=4` 的层，C4 部分作为 `extra_k_cache` / `extra_indices_in_kvcache` 传给 FlashMLA，不是替代完整 attention 的 KV，只是补充 SWA 之外的远距离信息。

**这意味着：GLM HiSparse 管的是"主 attention 的全部 KV 来源"；DSV4 C4 HiSparse 只管"复合 attention 中 C4 分支的 KV 来源"，SWA 和 C128 分支不受影响。**

#### 5.9.3 Indexer 的输入空间不同

GLM/NSA indexer 面向**原始序列 token**。`NSAIndexerMetadata.topk_transform()` 把 logits 转成 page/table indices，top-k 大小来自 `config.index_topk`（通常 2048）（`nsa_backend.py:226-239`）。

DSV4 C4 indexer 面向**C4 compressed 序列**：

```text
c4_seq_lens = seq_lens // 4  (metadata.py:47)
c4_page_size = page_size // 4
top-k = 512 compressed tokens = 2048 原始 tokens
```

所以 GLM top-k 的候选集合是原始 token（最多 128K 个）；C4 top-k 的候选集合是 compressed C4 item（最多 32K 个，因为 128K/4=32K）。

#### 5.9.4 Top-k 复用策略不同

GLM/NSA 每层都有 indexer slot（`get_num_indexer_layers` 返回 `num_hidden_layers`，`model_config.py:148-149`），但支持 `skip_topk`：部分层可以复用前一层的 top-k 结果，由 `index_topk_freq` 或 `index_topk_pattern` 控制（`deepseek_v2.py:1429-1447`）。

DSV4 的 indexer 层数 = `sum(1 for r in compress_ratios if r == 4)`（`model_config.py:150-152`）。只有 `compress_ratio == 4` 的层有 C4Indexer，其他层根本不做 C4 sparse attention。这不是"skip"，是结构性地只在特定层启用。

#### 5.9.5 HiSparse swap-in 的接入点不同

GLM/NSA：在 attention backend 内部做 swap-in。flow 是 indexer → topk → **进入 attention backend** → swap-in → sparse attention：

```python
# nsa_backend.py:1612-1618
page_table_1 = hisparse_coordinator.swap_in_selected_pages(
    req_pool_indices, seq_lens, topk_indices, layer.layer_id)
```

DSV4 C4：在 indexer 生成 raw indices 后**立即**做 swap-in，还没进入 attention backend：

```python
# dsv4/indexer.py:446-458
core_metadata.c4_sparse_page_indices = hisparse_coordinator.swap_in_selected_pages(
    req_pool_indices=forward_batch.req_pool_indices,
    compressed_seq_lens=indexer_metadata.c4_seq_lens,
    top_k_result=raw_indices,
    layer_id=compress_layer_id)
```

注意 DSV4 传的是 `compressed_seq_lens=c4_seq_lens`（已除以 4），而 NSA 传的是原始 `seq_lens`。

#### 5.9.6 Attention backend 消费 indices 的方式不同

GLM/NSA 有多种 sparse backend：`flashmla_sparse`、`flashmla_kv`、`fa3`、`tilelang`。以 `flashmla_sparse` 为例，`page_table_1` 直接作为唯一的 KV indices：

```python
indices_input = page_table_1.unsqueeze(1)  # [num_tokens, 1, topk]
o, _, _ = flash_mla_sparse_fwd(q=q_input, kv=kv_cache, indices=indices_input, ...)
# nsa_backend.py:1766-1771
```

DSV4 的 FlashMLA 调用有**双层 indices**：

```python
o = flash_mla.flash_mla_with_kvcache(
    indices=swa_page_indices,              # SWA 部分（滑动窗口，不走 HiSparse）
    extra_k_cache=extra_k_cache,           # C4 KV buffer
    extra_indices_in_kvcache=extra_indices, # C4 HiSparse 输出的 device locs
    extra_topk_length=extra_topk_lengths)
# deepseek_v4_backend.py:1001, 1036
```

GLM 的 sparse indices 指向唯一的 `kv_cache`；DSV4 的 C4 sparse indices 指向 `extra_k_cache`（一个独立于 SWA cache 的 buffer）。

#### 5.9.7 总结表

| 维度 | GLM/NSA Sparse Attention | DSV4 C4 Sparse Attention |
|------|--------------------------|--------------------------|
| attention 角色 | 该层的主（唯一）attention | 复合 attention 中的一个 extra 分支 |
| indexer 候选空间 | 原始 token 序列（128K） | C4 compressed 序列（32K） |
| top-k 数量 | 2048 raw tokens | 512 compressed tokens (= 2048 raw) |
| indexer 层分布 | 每层有 slot（可 skip） | 只有 compress_ratio==4 的层 |
| HiSparse 接入点 | attention backend 内部 | indexer 输出后、进入 backend 前 |
| indices 语义 | 主 KV indices（唯一来源） | extra KV indices（补充 SWA） |
| HiSparse offload 范围 | 全部 MLA KV | 仅 C4 KV（SWA/C128/indexer 不 offload） |
| seq_lens 给 swap-in | 原始 seq_lens | compressed seq_lens (÷4) |

**一句话总结：GLM Sparse Attention 是"对原始 token 做 NSA top-k，然后主 attention 只读这些 KV"；DSV4 C4 是"先压缩生成 C4 KV，再在 SWA + extra sparse 复合 attention 中把 C4 当作 extra sparse KV 使用"（每层只有一种 extra 类型：C4 或 C128）。二者在角色、indexer 作用域、层分布、top-k 空间、HiSparse 接入点、attention kernel 参数语义 6 个维度上都不同。**

## 第一部分：SGLang HiSparse 能力解读

先看对象：Host 存全量 KV，Device 只保留固定大小的 hot buffer。每步 decode 时，indexer 选出 top-k token；若某个 token 的 KV 不在 hot buffer，swap-in 把它从 Host 搬到 Device，满了就按 LRU 驱逐旧 KV。

本文讲 GPU 参考实现：为什么显存从 O(seq_len) 降到 O(device_buffer_size)，请求生命周期怎么走，swap-in kernel 如何输出 FlashMLA 可直接消费的 `indices`。

### 1.1 HiSparse 解决什么问题

长序列 decode 的痛点不是 sparse attention 算不动，而是 KV cache 仍全量常驻 GPU。HiSparse 的 before/after 是：

| | KV 存放 | attention 读到的 KV |
|---|---|---|
| Baseline | 全量 KV 在 GPU | indexer 选 top-k，attention 从全量 KV 里读 |
| HiSparse | 全量 KV 在 CPU pinned memory，GPU 只留 hot buffer | indexer 选 top-k，swap-in 先保证 top-k 在 hot buffer，再 attention |

长序列 decode 时，全量 KV cache 常驻 GPU 显存太贵。以 DeepSeek-V3.2（MLA, kv_lora_rank=512, qk_rope_head_dim=64）为例：

```
单请求 128K tokens 的 KV 显存 ≈ 128K × (512+64) × 2bytes × 61layers ≈ 8.5 GB
```

一个请求约 8.5 GB，几十并发会直接吃满显存。HiSparse 把 GPU 侧 KV 从 O(seq_len) 改成 O(device_buffer_size)：例如每步 `top_k=2048`、`device_buffer_size=4096`，GPU 只保留 4096 个 LRU slot 加 1 个 newest reserved slot。`device_buffer_size` 通常是 top-k 量级，但不是严格等于 `top_k`。

### 1.2 资源模型

```
启用 HiSparse:
  GPU KV 显存 = max_num_reqs × (device_buffer_size + 1) × kv_dim × num_layers
  Host KV 内存 = max_num_reqs × max_seq_len × kv_dim × num_layers / compress_ratio

不启用 HiSparse:
  GPU KV 显存 = max_num_reqs × max_seq_len × kv_dim × num_layers
```

GPU 公式里没有 `max_seq_len`，这是关键；代价是 Host KV 仍按序列长度增长。当 `max_seq_len=128K, device_buffer_size=4096` 时，GPU KV 约省 **30x**。上式的 Host KV 指被 HiSparse 下沉的 KV pool：NSA/DSA 是全部 MLA KV；DSv4 只套 C4 pool，SWA/C128/indexer pool 不在这个公式里。

### 1.3 适用范围

当前口径要分三层：

| 来源 | 范围 / 约束 |
|---|---|
| 官方 Guide | DSA 架构示例包括 DeepSeek-V3.2、GLM-5.1；另支持 DeepSeek V4。当前要求 **PD disaggregation mode**，只在 **decode instance** 启用；prefill instance 不需要 `--enable-hisparse` |
| 本地 SGLang 源码 | `is_deepseek_dsa()` 包含 `DeepseekV3ForCausalLM`、`DeepseekV32ForCausalLM`、`MistralLarge3ForCausalLM`、`PixtralForConditionalGeneration`、`GlmMoeDsaForCausalLM`；`is_deepseek_v4()` 包含 `DeepseekV4ForCausalLM`、`DeepseekV4ForCausalLMNextN` |
| Roadmap | 计划覆盖 MiniMax-M3、DeepSeek-V4、GLM-5.2、GLM-5.1；生产方向包括 1MB context、MTP、TwoBatch Overlap、Decode L2 RadixTree/HiCache offloading、NIXL DRAM KV、TP rank CPU memory sharing、Quest 等扩展 |

其他约束：decode 侧当前要关闭 radix cache；NSA/DSA 路径 `compress_ratio=1`，DSv4 HiSparse 下沉的是 C4 pool，C4 的 `compress_ratio=4`；BF16 KV 走 `flashmla_sparse`，FP8 KV 走 `flashmla_kv`，DeepSeek V4 走独立 `dsv4` backend。

### 1.4 上游资料入口

| 资料 | 链接 | 读它解决什么问题 |
|---|---|---|
| HiSparse Roadmap | [sglang#28874: HiSparse Roadmap for Long-Context Sparse Serving](https://github.com/sgl-project/sglang/issues/28874) | 模型支持、1MB 长上下文优化、PD decode 优化、HiCache/offloading 兼容等后续方向 |
| 官方文档 | [HiSparse: Hierarchical Sparse Attention](https://lmsysorg.mintlify.app/docs/advanced_features/hisparse_guide) | 启用条件、`--enable-hisparse`、`--hisparse-config`、PD decode 部署方式 |
| 官方博客 | [HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) | 为什么要做 hierarchical memory、hot buffer + host KV pool 的设计直觉 |

---

## 第二部分：HiSparse 架构分析

### 2.1 数据结构总览

下图把 HiSparse 拆成三个对象：Host KV Pool 存全量 KV；Device Hot Buffer 存当前可被 attention 读取的 KV；Indexer KV Cache 体积小，仍全量常驻 GPU，用来选 top-k。

关键映射：
- `req_to_host_pool[req, pos] -> host_slot`：某个请求的某个 token 在 Host pool 的位置。
- `req_device_buffer_tokens[layer, req, slot] -> token_pos`：hot buffer 的某个 slot 当前放了哪个 token。
- `req_device_buffer_token_locs[layer, req, slot] -> loc`：hot buffer slot 对应的 device KV 物理位置。
- `lru_slots[layer, req, :]`：每层每请求的 LRU 顺序。

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

一个请求从进入到结束走四步：Prefill 先写全量 device KV；Staging 把全量 KV 备份到 Host 并释放大块 device KV；Decode 每步只维护 hot buffer；Finish 清理 Host/Device 状态。

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

> **为什么 Staging 必须先备份再释放？** Prefill 结束时，全量 KV 只在 device pool 里。必须等 D2H 全量备份完成，Host pool 有完整副本后，才能释放 device pool 的大块槽位。HiCache 是复制，device 数据仍保留；HiSparse 是迁移，备份完成后释放 device 上的全量 KV。

### 2.3 prefill 长度 vs hot buffer 大小的两种场景

**核心问题**：prefill 长度是否超过 `device_buffer_size`。这决定 decode 是否需要从 Host swap-in。

- **短序列**：prefill 放得进 hot buffer，decode 全程命中。
- **长序列**：prefill 放不进 hot buffer，hot buffer 作为 LRU 缓存使用。

设 `device_buffer_size`（hot buffer 容量，典型 4096）。staging 时 `alloc_device_buffer` 按 prefill 长度走两条分支（`hisparse_coordinator.py:288-333`、`hisparse_memory_pool.py:240-276`）。

两个尺寸常量：

- `device_buffer_size`：hot buffer 标称容量，decode 时的 LRU 缓存大小。
- `padded_buffer_size = device_buffer_size + page_size`：实际分配量，多出一页作为当前新 token 的 reserved slot。

**场景 A：prefill ≤ device_buffer_size（短序列）**

整个序列装得进 hot buffer，**decode 全程命中、无需 swap-in、不读 host**。

| 步骤 | 行为 |
|------|------|
| staging 分配 | `alloc_size = 页对齐(prefill_len)`，只分配实际需要量，不浪费到满（`:297-300`） |
| device 留存 | 全部 prefill token 留在 device，全 resident |
| decode | swap-in kernel fast path 直接返回 device locs，不读 host |
| 序列增长 | 每步若 `seq_len` 超当前容量，`_grow_device_buffers` 页对齐扩容，直到触顶 `device_buffer_size`（`:335-406`） |
| 触顶 | `seq_len` 达 `device_buffer_size` 后分配量升到 `padded_buffer_size`，转入场景 B 行为 |

**场景 B：prefill > device_buffer_size（长序列）**

序列装不下，**必须 swap-in**，hot buffer 当 LRU 缓存用。

| 步骤 | 行为 |
|------|------|
| staging 分配 | `alloc_size = padded_buffer_size`（buffer + reserved page，`:301-302`） |
| device 留存 | 从全量槽位中**保留前 `device_buffer_size` 个**作初始内容，其余 `free_hisparse_indices` 释放（`hisparse_memory_pool.py:250-251`） |
| 初始标记 | `device_buffer_tokens = [0,1,...,buffer_size-1]`（`:328-330`），告诉 kernel buffer 初始装 token 0~N |
| decode | 每步 indexer 选 top-k → swap-in 判 hit/miss → miss 从 host 搬入、LRU 踢旧 → 输出 device locs |
| 新 token | 每步新 token 写 reserved slot（`map_last_loc_to_buffer:458-460`），异步增量备份到 host |

**为什么初始保留"前 N 个"而非"最重要 N 个"**：staging 时 decode 未开始、无 query，无法判断 token 重要性，故按序列顺序取前 N。这只是 LRU 缓存初始值，decode 第一步 indexer 即按真实 query 选 top-k，miss 的从 host 换入——一两步内 buffer 内容就被实际需要的 token 替换。

**冷启动代价**：首步 decode 的 top-k 与"前 N"初始内容大概率不重合（尤其序列尾部 token，局部性最强却不在 buffer），首步 miss 较多。因相邻 decode 步 top-k 高度重叠，一两步后命中率回升至稳态。

### 2.4 Decode 每步详细流程

每生成一个 token，先在 `prepare_for_decode` 做状态更新：备份上一步 newest token，并为本步 newest token 分配 reserved slot。随后 `forward_decode` 逐层执行：写本层 KV、indexer 选 top-k、swap-in 补齐 missing KV、attention 读取 hot buffer。

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

### 2.5 swap-in 机制（核心 CUDA kernel）

每步 decode，indexer 输出 top-k logical token positions；attention 需要的是这些 token 在 hot buffer 里的 physical device locations。swap-in kernel 做三件事：判定 hit/miss，miss 时从 Host 搬入并更新 LRU，最后输出 `top_k_device_locs`。

先澄清一个容易误解的点：HiSparse 自定义的 CUDA kernel 主要是 **residency / swap-in kernel**，不是重新实现 attention 数学。attention 计算复用已有的 FlashMLA sparse / FlashMLA KV / tilelang / FA3 等 backend；HiSparse 做的是把 indexer 产生的 logical top-k token 转换成 hot buffer 中的 physical device locations，再交给这些已有 attention kernel 的 `indices` / `page_table` 类接口。

关键设计是：**一个 CUDA kernel 内完成 hit/miss 判定、LRU 更新和 Host→Device copy**：

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

#### 2.5.1 源码实现细节（经逐项验证）

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

FlashMLA 接受 `indices` / `indices_in_kvcache`，按 encoded KV index 读 `k_cache`；HiSparse swap-in kernel 输出的 `top_k_device_locs` 正好是同一类 index。关键契约如下：

| 问题 | 事实 |
|---|---|
| FlashMLA index 是什么 | 测试里构造 `block_table_cpu[i, cur_abs_indices // block_size] * block_size + (cur_abs_indices % block_size)`；kernel 内把 `token_index` / `my_indices[i]` 拆成 `block_index` / `block_idx` 与 `rel_idx_in_block` / `idx_in_block` |
| sparse 路径是否查 `block_table` | 不查；sparse 路径里的 `block_table` 是兼容保留参数，`flash_mla_interface.py:77` 注明 sparse attention 时可为 None |
| HiSparse 为什么能直接接 | hot buffer 不是另开的 tensor，而是 KV cache tensor 的连续区域；swap-in 写入的 `device_buffer[slot]` 与 FlashMLA 读取的 `k_cache` 在同一地址空间 |
| 形状怎么对齐 | FlashMLA `indices` 是 `[batch, seq_len_q, topk]`，decode 时 `seq_len_q=1`；`top_k_device_locs [batch, topk]` 做 `unsqueeze(1)` |

FlashMLA 测试中的 encoded index 构造：

```python
# test_flashmla.py line 447-468
cur_blocked_indices = block_table_cpu[i, cur_abs_indices // block_size] * block_size + (cur_abs_indices % block_size)
indices_in_kvcache[i, j, :] = cur_blocked_indices
```

`nsa_backend.py:1612-1626` 进一步证明：三条分支都产出同一个 `page_table_1`，下游 backend 不区分它来自默认 page table、fused topk 还是 HiSparse。

```python
if forward_batch.hisparse_coordinator is not None:
    # 开 HiSparse：swap-in kernel 直接产出 FlashMLA-style encoded index
    page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(
        forward_batch.req_pool_indices, forward_batch.seq_lens, topk_indices, layer.layer_id,
    )
elif envs.SGLANG_NSA_FUSE_TOPK.get():
    # fuse 路径：topk 已经是 FlashMLA 可消费的 encoded index，直接用
    page_table_1 = topk_indices
else:
    # 默认两级寻址：logical 位置过 block_table 翻译成 FlashMLA-style encoded index
    page_table_1 = transform_index_page_table_decode(
        page_table=metadata.page_table_1, topk_indices=topk_indices, page_size=1,
    )

# 下面所有 backend 都只认这个 page_table_1，不关心它是哪条分支产出的：
if self.nsa_decode_impl == "flashmla_sparse":
    return self._forward_flashmla_sparse(..., page_table_1=page_table_1, ...)
elif self.nsa_decode_impl == "flashmla_kv":
    return self._forward_flashmla_kv(..., page_table_1=page_table_1, ...)
elif self.nsa_decode_impl == "tilelang":
    return self._forward_tilelang(..., page_table_1=page_table_1, ...)
# fa3 同理
```

结论：HiSparse 在 GPU 上只是把 `indices_in_kvcache` 的生产者从 Python page-table translation 换成 `swap_in_selected_pages`，FlashMLA 算子本身不改。注意 `page_size=1` 与是否查表是两件事：默认分支即使 `page_size=1`，仍走 logical → block_table → encoded index。

> **对 NPU 的提示**：GPU 不改 FlashMLA 依赖两点：hot buffer 复用 KV tensor 地址空间，attention 算子接受 encoded KV index（已经编码好的物理 KV 行号）。当前 GLM-5 NPU 生产调用是 `layout_query="TND"` + `layout_kv="PA_BSND"` + `block_table`，SFA 收到 logical S2，不是 FlashMLA 绝对物理槽号。因此 NPU 主线应先保持 PA_BSND，通过重映射 `block_table` 读 hot buffer；BSND 一层寻址分支存在，但要切 query/rope/output，且 pytest/eager golden 只覆盖 BSND K=16/32，GLM-5 K=2048 生产参数组合仍须 golden 对齐测试。方案 C（改 SFA）只作 A/B 都不通过时的最后备选。

**DSv4 路径的 page_size 差异**

NSA 路径（DeepSeek V3.2）的 hot buffer 管理粒度是 1 token（`page_size=1`）。DSv4 路径因为 C4 `compress_ratio=4`，管理粒度是 1 compressed item（代表 4 个原始 token）。严格来说 DSv4 C4 的 residency 粒度是固定 `device_buffer_size` 个 compressed items，而不是 raw tokens，但量级不变。

### 2.6 Stream / 同步模型

搬运（D2H 备份）和计算（forward）不能都挤在一条 stream 上。HiSparse 用三条 stream 分工，再用 event 保证依赖顺序：

- **Main Stream**：执行 prefill / decode 计算。
- **write_staging_stream**：staging 时做一次性全量 D2H 备份，可与其他 batch 的计算并行。
- **decode_backup_stream**：decode 时备份上一步 newest token，不阻塞主线计算下一步。

下图竖排三条流，箭头是 event 同步。关键约束只有一条：**读或覆写某份 KV 前，必须确认相关搬运已完成**。

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

### 2.7 Indexer 与 HiSparse 的关系

容易混淆的一点：**indexer 不是 HiSparse 新增的**。它是 DSA 模型自带模块，负责每层选出 top-k token。HiSparse 复用这个输出：indexer 决定本层要读哪些 token，HiSparse 决定这些 token 的 KV 在 device 还是 Host、miss 时搬到哪个 hot buffer slot。

Indexer 是 DSA 模型架构的一部分（模型训练时就有），HiSparse 是推理优化。两者关系：

- **Indexer**：每层计算 attention score，选出 top-k 个最重要的 token positions。这是模型本身的 sparse attention 机制，不管有没有 HiSparse 都会运行。
- **HiSparse**：利用 indexer 的 top-k 输出，只把这些 token 的 KV 保留在 GPU 上。

不启用 HiSparse 时：indexer 选出 top-k → page table 翻译 → attention 从全量 KV cache 中读取 top-k 位置。
启用 HiSparse 时：indexer 选出 top-k → swap-in kernel 把 miss 的从 host 拉到 hot buffer → attention 从 hot buffer 读取。

关键区别：
- 不启用时，`topk_indices` 经过 `transform_index_page_table_decode` 翻译成 physical locs
- 启用时，强制 `force_unfused_topk=True`（indexer 返回 raw logical positions），然后由 swap-in kernel 返回 physical device buffer locs

### 2.8 两条路径对比（简表）

| | NSA 路径 (DeepSeek V3.2, GLM-5) | DSv4 路径 (DeepSeek V4) |
|---|---|---|
| compress_ratio | 1 | C4 pool = 4 |
| index_topk | 2048 tokens | 512 compressed tokens (= 2048 原始 tokens) |
| 备份频率 | 每步 | 每 4 步（compress_ratio 边界） |
| Attention backend | `flashmla_kv` / `flashmla_sparse` / tilelang | `flash_mla.flash_mla_with_kvcache`（SWA indices + extra_indices 双路） |
| `_grow_device_buffers` | 有（短序列动态增长） | 无（固定 padded_buffer_size） |

完整对比（含 pool 类名、kernel 名称、indexer 差异、FlashMLA 接口语义）见 [03-nsa-vs-dsv4-differences.md § 3.2](03-nsa-vs-dsv4-differences.md#32-sglang-gpu-侧两条-hisparse-路径对比)。

### 2.9 配置参数

官方 Guide 的当前部署口径是 PD disaggregation：prefill instance 不启 HiSparse，decode instance 才启 `--enable-hisparse`。Blog 也讨论 co-located 形态；本文把四阶段 lifecycle 作为 GPU 机制说明，生产部署以 Guide/Roadmap 为准。

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --disaggregation-mode decode \
  --disable-radix-cache \
  --enable-hisparse \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10}'
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `top_k` | 2048（或 model config 的 `index_topk`） | 每步每层选多少 token |
| `device_buffer_size` | `2 × top_k` | hot buffer 大小，必须 >= top_k |
| `host_to_device_ratio` | 2 | 逻辑池大小 = device pool × ratio，决定最大序列长度 |

上表是本地源码默认值（`srt/mem_cache/sparsity/factory.py`）；官方部署示例使用 `device_buffer_size=6144, host_to_device_ratio=10`。Guide 建议按 host 内存配 `host_to_device_ratio`：约 1 TB 可用 host 内存取 5，约 2 TB 取 10。

### 2.10 关键源文件

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

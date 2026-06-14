## 第一部分：SGLang HiSparse 能力解读

> **先用一个比方建立直觉（全文都会用到）**
>
> 把 GPU 显存想象成一张**小书桌**，把 host 内存想象成旁边的**大仓库**。
>
> - 模型推理（attention 计算）只能在书桌上做，但书桌很小，放不下一本几百页的书（长序列的全量 KV）。
> - 于是我们把整本书放进仓库，书桌上只摊开"接下来要用到的那几页"。
> - 每写一个新字（decode 一步），模型先判断"这一步要参考书里的哪几页"（indexer 选 top-k），把不在桌上的页从仓库搬上来（swap-in），桌子满了就把最久没看的页放回仓库（LRU 淘汰）。
>
> 这张"小书桌"就是下文反复出现的 **hot buffer**，"大仓库"就是 **host KV pool**，"把页搬上桌"就是 **swap-in**，"放回最久没看的页"就是 **LRU 淘汰**。记住这个画面，后面的公式和代码都是在精确地描述它。

### 1.1 HiSparse 解决什么问题

**一句话**：长序列推理时，KV cache 大到 GPU 显存装不下，HiSparse 让 GPU 只放一小桌、其余放仓库，按需搬运，从而用很小的显存跑很长的序列。

长序列 decode 时，全量 KV cache 常驻 GPU 显存太贵。以 DeepSeek-V3.2（MLA, kv_lora_rank=512, qk_rope_head_dim=64）为例：

```
单请求 128K tokens 的 KV 显存 ≈ 128K × (512+64) × 2bytes × 61layers ≈ 8.5 GB
```

一个请求就要 8.5 GB，几十个并发请求直接把显存撑爆——这就是要解决的痛点。

HiSparse 的核心能力：**只在 GPU 上保留一个固定大小的 "hot buffer"（就是那张小书桌，本质是个 LRU 缓存），全量 KV 放在 pinned host memory（大仓库），每步 decode 按 indexer 选出的 top-k token 做 swap-in（把要用的页搬上桌），GPU KV 显存从 O(seq_len) 降到 O(top_k)。** 也就是说，显存占用不再随序列变长而增长，只取决于"每步看几页"。

### 1.2 资源模型

直观地说：开了 HiSparse，GPU 上每个请求只占"一张固定大小的书桌"（与序列长度无关）；不开，则要占"整本书那么大"（随序列线性增长）。

```
启用 HiSparse:
  GPU KV 显存 = max_num_reqs × (device_buffer_size + 1) × kv_dim × num_layers
  Host KV 内存 = max_num_reqs × max_seq_len × kv_dim × num_layers / compress_ratio

不启用 HiSparse:
  GPU KV 显存 = max_num_reqs × max_seq_len × kv_dim × num_layers
```

注意启用时 GPU 那行没有 `max_seq_len`——这正是关键：**显存成本和序列长度脱钩了**。代价是全量 KV 挪到了 host 内存（host 便宜、量大）。

当 max_seq_len=128K, device_buffer_size=4096 时，GPU KV 省约 **30x**。

### 1.3 适用范围

- 仅支持 DSA 模型（DeepSeek-V3.2、GLM-5、DeepSeek V4、MistralLarge3、Pixtral）
- 必须 `--disable-radix-cache`
- 两条路径：NSA 路径（compress_ratio=1）和 DSv4 路径（compress_ratio=4）
- BF16 KV → `flashmla_sparse` backend；FP8 KV → `flashmla_kv` backend

---

## 第二部分：HiSparse 架构分析

### 2.1 数据结构总览

下图把"书桌 + 仓库"画成实际的数据结构。**仓库在上（Host）**，存整本书；**书桌在下（Device/GPU）**，只摊开几页。中间那条向上的箭头是"看完的页随手归档进仓库"（D2H backup）。

读图小贴士：
- `Host KV Pool` = 仓库货架本身，`req_to_host_pool` = 货架索引（"这个请求的第几页放在仓库哪个格子"）。
- `Hot Buffer` = 书桌，`req_device_buffer_tokens` 等三张跟踪表 = 桌上便签（"此刻桌上摊的是哪几页、在桌面哪个位置、谁最久没碰"）。
- `Indexer KV Cache` 是个例外：它体积很小，所以**整本都常驻 GPU**，专门用来快速判断"下一步该看哪几页"。

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

一个请求从进来到结束走四步。用书桌比方：**先在大桌上写完开头（Prefill）→ 把整本书归档进仓库、换成小书桌（Staging）→ 边写边按需翻页（Decode）→ 收工清桌（Finish）**。

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

> **为什么 Staging 要先"归档整本书再换小桌"，不能边写边换？** Prefill 阶段全量 KV 还在大书桌上，必须等它**完整复制进仓库**之后，才能放心把大书桌收掉换成小书桌——否则还没归档的页就丢了。这一步是"复制 + 删除"的原子动作：仓库里有了完整副本，GPU 上的大块显存才被释放。（这也是 HiSparse 与 HiCache 的关键差别：HiCache 是纯复制，device 上的数据会保留；HiSparse 是迁移，复制完就删 device。）

### 2.3 prefill 长度 vs hot buffer 大小的两种场景

**核心问题**：整本书的页数（prefill 长度），到底比书桌大还是比书桌小？这决定了要不要翻页。

- **书短，整本能摊在桌上** → 根本不用去仓库翻页，全程命中（场景 A）。
- **书长，桌上摊不下** → 必须按需翻页，书桌退化成一个"最近常看的几页"的缓存（场景 B）。

设 `device_buffer_size`（hot buffer 容量，典型 4096）。staging 时 `alloc_device_buffer` 按 prefill 长度走两条分支（`hisparse_coordinator.py:288-333`、`hisparse_memory_pool.py:240-276`）。

两个尺寸常量：

- `device_buffer_size`：hot buffer 标称容量（书桌能摊几页），decode 时 swap-in LRU 缓存大小
- `padded_buffer_size = device_buffer_size + page_size`：实际分配量，多出一页是写**当前新生成 token** 的 reserved slot（书桌上专门留的一块"草稿区"，写正在生成的那个字）

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

每生成一个 token，做两件事：**先收拾桌面**（`prepare_for_decode`：把上一步写的字归档进仓库，给这一步的新字留好草稿位），**再逐层写字**（`forward_decode`：每一层都问 indexer "这层要看哪几页" → 把缺的页搬上桌 → 在这几页上算 attention）。下面的伪代码就是这两步的展开。

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

> **这一节是 HiSparse 的心脏。** 用书桌比方：每步 decode，indexer 给出一张"这层要看的页码清单"（top-k logical positions），但模型真正要的是"这些页此刻摊在桌面的哪个物理位置"。swap-in kernel 干的就是这个翻译 + 搬运：**对清单上每一页，在桌上找到 → 找不到就从仓库搬上来并挤掉一页最久没看的 → 最后输出每一页在桌面的物理坐标。** 整个过程在一个 CUDA kernel 内一气呵成，不回 Python，也不劳烦 CPU。

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

> **一句话**：FlashMLA 本来就是"给我一张行号清单，我去 KV tensor 里读这几行"的算子。HiSparse 的 swap-in kernel 输出的恰好就是这样一张行号清单，所以直接对接、无需改动。

拆开讲，关键是搞清楚两件事：FlashMLA 想要什么输入，HiSparse 给的是什么。

**① FlashMLA 想要什么**：它是为 sparse attention 设计的，本来就不读全量 KV，而是接受一个 `indices` 参数——含义是"**到 KV cache tensor 里，按这些行号去读**"。它不关心这些行号是怎么算出来的，只负责照着地址读、算 attention。

不开 HiSparse 时，这张行号清单是这么来的：indexer 给出 logical 位置 → 过 page table 翻译成 physical 行号 → 喂给 FlashMLA。

**② HiSparse 给的是什么**：这里有个**容易忽略但决定性的实现细节——hot buffer 不是另开的一块 tensor，它就是 KV cache tensor 里的一段连续区域**。所以 swap-in kernel 把仓库的页搬上书桌时，写入的位置 `device_buffer[slot]` 本身就是 KV cache tensor 内的真实行。它输出的 `top_k_device_locs`，就是这些行在 tensor 里的 physical 行号。

于是两边的"行号"说的是同一件事——都是"KV cache tensor 内的 physical row offset"。FlashMLA 测试里 `indices` 也正是这么构造的：

```python
# FlashMLA 测试中 indices 的构造方式（test_flashmla.py line 447-468）：
cur_blocked_indices = block_table_cpu[i, cur_abs_indices // block_size] * block_size + (cur_abs_indices % block_size)
indices_in_kvcache[i, j, :] = cur_blocked_indices
```

**③ 对接**：剩下只差形状对齐。FlashMLA `indices` 是 `[batch, seq_len_q, topk]`（decode 时 `seq_len_q=1`），HiSparse 的 `top_k_device_locs` 是 `[batch, topk]`，`unsqueeze(1)` 补一维即可。kernel 拿到行号直接按地址读，不做任何额外翻译。

**所以"不需要改 FlashMLA"的本质**：HiSparse 没有改变 FlashMLA 的输入契约，只是把"行号清单的生产者"从 page table 换成了 swap-in kernel——而且因为 hot buffer 复用了 KV tensor 的地址空间，生产出来的行号语义完全一致。这正是 GPU 方案能"只写一个 swap-in kernel 就搞定整个 HiSparse"、不动任何 attention 算子的根本原因。

**代码铁证**（`nsa_backend.py:1612-1626`）：行号清单 `page_table_1` 由一个**三选一**的分支产出，三条路算出的是**同一个变量**，随后**不加区分地**喂给所有 attention backend。

```python
if forward_batch.hisparse_coordinator is not None:
    # 开 HiSparse：swap-in kernel 直接产出 physical 行号
    page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(
        forward_batch.req_pool_indices, forward_batch.seq_lens, topk_indices, layer.layer_id,
    )
elif envs.SGLANG_NSA_FUSE_TOPK.get():
    # fuse 路径：topk 已经是 physical，直接用
    page_table_1 = topk_indices
else:
    # 默认两级寻址：logical 位置过 block_table 翻译成 physical 行号
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

三点结论直接从这段代码读出：

1. **算子不知道也不关心行号是谁生产的**：三分支产出物变量名相同、下游调用相同，FlashMLA 只是收到一张 `page_table_1` 就照着读。
2. **HiSparse 做的就是"换生产者"**：它用 `swap_in_selected_pages`（一级，直接出 physical 行号）替换掉默认的 `transform_index_page_table_decode`（两级，logical → 查 block_table → physical），算子本身一行不改。
3. **page_size 与寻址级数无关**：默认分支显式传 `page_size=1` 仍走两级翻译——印证了"一格放一个 token"和"要不要查表翻译"是两件独立的事。

> **对 NPU 适配的提示**：这个"零修改"红利完全建立在 ① hot buffer 与 KV tensor 共用地址空间、② 算子接受 physical 行号清单 这两个前提上。NPU 的 SFA 算子若用的是 logical 索引 + block_table 两级寻址（而非直接吃 physical 行号），就享受不到这个红利——这正是 02 文档里"路线 C 要改算子内部"的由来。

**DSv4 路径的 page_size 差异**

NSA 路径（DeepSeek V3.2）的 hot buffer 管理粒度是 1 token（`page_size=1`）。DSv4 路径因为 `compress_ratio=4`，管理粒度是 1 compressed item（代表 4 个原始 token）。严格来说 HiSparse 的 residency 粒度是 O(top_k compressed items) 而非 O(top_k raw tokens)，但量级不变。

### 2.6 Stream / 同步模型

直觉先行：搬运（D2H 备份）和计算（forward）如果都挤在一条流水线上会互相等。所以 HiSparse 用**三条并行的流水线**分工，再用 event 当"红绿灯"保证先后顺序：

- **Main Stream**：主线，跑 prefill / decode 的真正计算。
- **write_staging_stream**：staging 时把整本书归档进仓库（一次性全量 D2H），和别的 batch 的计算并行跑。
- **decode_backup_stream**：decode 时把上一步那个新字归档进仓库（每步增量 D2H），不挡主线算下一步。

下图竖排三条流，箭头是 event 同步（`record` = 亮灯，`wait` = 等灯）。关键约束只有一条：**用到某份数据前，必须确认它的搬运已完成**（比如 decode 主线读 KV 前要 `wait` 备份完成的 event）。

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

容易混淆的一点：**indexer 不是 HiSparse 发明的**。它本来就是 DSA 模型自带的零件，负责"每层挑出最该看的 top-k 个 token"。HiSparse 只是搭它的便车——既然模型自己已经算出"这步只看这几页"，那干脆就只把这几页留在 GPU 上。一句话：**indexer 决定看哪几页，HiSparse 决定这几页怎么放、从哪搬。**

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
| compress_ratio | 1 | 4 |
| index_topk | 2048 tokens | 512 compressed tokens (= 2048 原始 tokens) |
| 备份频率 | 每步 | 每 4 步（compress_ratio 边界） |
| Attention backend | FlashMLA sparse / tilelang | `flash_mla.flash_mla_with_kvcache`（SWA indices + extra_indices 双路） |
| `_grow_device_buffers` | 有（短序列动态增长） | 无（固定 padded_buffer_size） |

完整对比（含 pool 类名、kernel 名称、indexer 差异、FlashMLA 接口语义）见 [03-nsa-vs-dsv4-differences.md § 3.2](03-nsa-vs-dsv4-differences.md#32-sglang-gpu-侧两条-hisparse-路径对比)。

### 2.9 配置参数

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


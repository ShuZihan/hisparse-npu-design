## 第三部分：NSA 模型与 DeepSeek-V4 的 HiSparse 路径差异

> **TL;DR**：GLM-5/NSA 先做，DSV4 以后做。理由：NPU 侧已有完整 NSA/SFA 推理路径、compress_ratio=1 无压缩逻辑、swap-in layout 是 linear 的。公共层（Host pool、hot buffer、LRU、swap-in 调度）设计时参数化 `compress_ratio` 和 `layout_type`，未来接 DSV4 时只需实现模型接入层。

### 3.1 结论

GLM-5（及 DeepSeek-V3.2 等 NSA 模型）和 DeepSeek-V4 在 SGLang GPU 侧已经是两套独立的 HiSparse 实现。NPU 适配时也必须分开。

**公共层可以共享**：Host KV pool 管理、device hot buffer 分配、hit/miss/LRU 逻辑、固定 shape 输出 `top_k_device_locs`、swap-in 调度框架。

**模型接入层必须分开**：KV 压缩比、swap-in kernel layout、attention backend 接口、KV pool 结构完全不同。

### 3.2 SGLang GPU 侧两条 HiSparse 路径对比

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

### 3.3 NSA 路径的 HiSparse 端到端流程（GLM-5 / V3.2）

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

### 3.4 DSV4 路径的 HiSparse 端到端流程

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

### 3.5 NPU 适配的分模型策略

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

### 3.6 公共组件复用

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

### 3.7 GLM-5 在 SGLang 中的特化点

GLM-5（`GlmMoeDsaForCausalLM`）直接继承 `DeepseekV2ForCausalLM`，几乎没有 attention 相关的模型特化。在 NSA HiSparse 路径里，唯一的 GLM-5 特化是：

- **Head padding**：`nsa_backend.py:364`——GLM-5 的 `num_q_heads` 可能小于 16（例如 64 heads / TP8 = 8），需要 pad to 16 才能对齐 FlashMLA tile 要求
- **Blackwell workaround**：`server_args.py:1796`——GLM-5 在 Blackwell GPU 上禁用 MHA_ONE_SHOT

除此之外，GLM-5 和 DeepSeek-V3.2 在 HiSparse 路径里完全一致。**NPU HiSparse 方案不需要为 GLM-5 单独设计**，只要做好通用 NSA/SFA 路径的适配即可覆盖所有 `is_deepseek_nsa` 模型。

### 3.8 为什么不能共用同一个 KV pool 实现

结论：**可以共用 HiSparse residency runtime（LRU / hit-miss / host pool / hot buffer / swap-in 调度），不能共用同一个 KV pool 实现**。GLM/NSA 和 DSV4 C4 的 KV pool 绑定了 4 个契约，每一个都不同：

#### 3.8.1 token 语义不同

GLM/NSA 的 HiSparse pool：一个原始 token → 一个 KV slot。`topk_indices` 选的就是原始序列 token id。

源码证据：`HiSparseNSATokenToKVPool.translate_loc_from_full_to_compressed()` 直接 `return full_indices`（`hisparse_memory_pool.py:90`），说明没有压缩 token 空间。

DSV4 C4 pool：4 个原始 token → 1 个 C4 compressed KV。只有 `(full_index + 1) % 4 == 0` 的位置才产生 C4 KV。

源码证据：`HiSparseC4DevicePool.translate_loc_from_full_to_compressed()`（`deepseek_v4_memory_pool.py:199-202`）：

```python
mask = (full_indices + 1) % self.compress_ratio == 0
compressed_indices = full_indices[mask] // self.compress_ratio
```

同一个 `loc=100`，在 GLM pool 里是"第 100 个原始 token"，在 DSV4 C4 pool 里要先判断是否合法再映射到 compressed loc。

#### 3.8.2 内存 layout 不同

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

#### 3.8.3 写入接口不同

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

#### 3.8.4 attention 消费方式不同

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

#### 3.8.5 如果强行共用的后果

如果试图让 GLM 和 DSV4 C4 共用一个 KV pool 实现，最容易出错的四个点：

1. **`loc` 语义错**：同一个 loc 值在两种 pool 里含义不同
2. **copy stride 错**：584 bytes packed uint8 vs `kv_cache_dim * dtype.itemsize` 的步长计算完全不同
3. **packed layout 被当普通 tensor 读**：如果把 DSV4 的 uint8 packed data 当 BF16/FP8 tensor 解释，读出的 K/V 值无意义
4. **attention kernel 收到错误格式的 KV**：FlashMLA sparse 期望的 `kv_cache` shape 和 DSV4 backend 期望的 `extra_k_cache` shape 完全不同

### 3.9 GLM Sparse Attention 与 DSV4 C4 的结构性差异（超越压缩比）

3.8 节说明了 KV pool 层面的 4 个契约差异。本节从 attention 子系统角度补充 6 个结构性差异，说明二者不只是"C4 有压缩、GLM 没有"，而是在整个 attention 架构中扮演的角色、indexer 语义、HiSparse 接入位置都不同。

#### 3.9.1 不是同一个 attention 子系统

GLM/DSA 走 SGLang 的 NSA 路线。`GlmMoeDsaForCausalLM` 直接继承 `DeepseekV2ForCausalLM`（`glm4_moe.py:1483`），使用 `nsa_backend.py` 的 FlashMLA sparse / FA3 / tilelang。

DSV4 走独立的 `deepseek_v4_backend.py`。每层由 `config.compress_ratios[layer_id]` 决定 attention 类型：`0`（SWA only）、`4`（SWA + C4 sparse）、`128`（SWA + C128）（`deepseek_v4.py:186-192`）。

只有 `compress_ratio == 4` 的层才有 C4Indexer。

#### 3.9.2 Sparse attention 的角色不同

**GLM：sparse attention 是该层的主 attention。** Indexer 选出 top-k token 后，FlashMLA 只读这些 token 的 KV，这就是这一层的完整 attention 输出。

**DSV4 C4：sparse attention 只是复合 attention 的一个分支。** DSV4 每层有一个 `compress_ratio`（由 `config.compress_ratios[layer_id]` 指定），决定该层的 attention 组成：

```text
compress_ratio=0:   SWA only（滑动窗口，无 extra sparse）
compress_ratio=4:   SWA + C4 extra compressed sparse attention (top-512 compressed tokens)
compress_ratio=128: SWA + C128 compressed/global branch
```

三者是逐层互斥的（`deepseek_v4.py:191` `assert compress_ratio in [0, 4, 128]`；`deepseek_v4_backend.py:972-979` if/elif 分支）。HiSparse 只涉及 `compress_ratio=4` 的层，C4 部分作为 `extra_k_cache` / `extra_indices_in_kvcache` 传给 FlashMLA，不是替代完整 attention 的 KV，只是补充 SWA 之外的远距离信息。

**这意味着：GLM HiSparse 管的是"主 attention 的全部 KV 来源"；DSV4 C4 HiSparse 只管"复合 attention 中 C4 分支的 KV 来源"，SWA 和 C128 分支不受影响。**

#### 3.9.3 Indexer 的输入空间不同

GLM/NSA indexer 面向**原始序列 token**。`NSAIndexerMetadata.topk_transform()` 把 logits 转成 page/table indices，top-k 大小来自 `config.index_topk`（通常 2048）（`nsa_backend.py:226-239`）。

DSV4 C4 indexer 面向**C4 compressed 序列**：

```text
c4_seq_lens = seq_lens // 4  (metadata.py:47)
c4_page_size = page_size // 4
top-k = 512 compressed tokens = 2048 原始 tokens
```

所以 GLM top-k 的候选集合是原始 token（最多 128K 个）；C4 top-k 的候选集合是 compressed C4 item（最多 32K 个，因为 128K/4=32K）。

#### 3.9.4 Top-k 复用策略不同

GLM/NSA 每层都有 indexer slot（`get_num_indexer_layers` 返回 `num_hidden_layers`，`model_config.py:148-149`），但支持 `skip_topk`：部分层可以复用前一层的 top-k 结果，由 `index_topk_freq` 或 `index_topk_pattern` 控制（`deepseek_v2.py:1429-1447`）。

DSV4 的 indexer 层数 = `sum(1 for r in compress_ratios if r == 4)`（`model_config.py:150-152`）。只有 `compress_ratio == 4` 的层有 C4Indexer，其他层根本不做 C4 sparse attention。这不是"skip"，是结构性地只在特定层启用。

#### 3.9.5 HiSparse swap-in 的接入点不同

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

#### 3.9.6 Attention backend 消费 indices 的方式不同

具体代码对比见 [§ 3.8.4](#384-attention-消费方式不同)。核心差异总结：

- **GLM/NSA**：`page_table_1`（即 `top_k_device_locs`）直接作为 FlashMLA sparse 的**唯一 KV indices**，指向单一的 `kv_cache`。
- **DSV4 C4**：`top_k_device_locs` 作为 `extra_indices_in_kvcache`，指向独立于 SWA cache 的 `extra_k_cache`。SWA 部分用另一组 `indices`，不走 HiSparse。

**GLM 的 sparse indices 是主 KV indices；C4 的 sparse indices 是 extra KV indices。** 这是结构性差异：GLM HiSparse 管的是整个 attention 的 KV 来源；DSV4 C4 HiSparse 只管复合 attention 中 C4 分支的 KV 来源，SWA 和 C128 不受 HiSparse 影响。

#### 3.9.7 总结表

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

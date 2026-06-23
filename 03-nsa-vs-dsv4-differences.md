## 第三部分：NSA 模型与 DeepSeek-V4 的 HiSparse 路径差异

结论先行：**GLM-5/NSA 先做，DeepSeek-V4 后做**。NPU 已有 GLM-5 的 `npu_lightning_indexer + npu_sparse_flash_attention` 路径，`compress_ratio=1`，KV 管理更直接；DSV4 的 C4/C128/SWA 双路 attention 在 NPU 上是另一项大工程。

公共层可以抽成 `HiSparseResidencyRuntime`：一个模型无关的驻留管理层，负责 host pool、hot buffer、LRU 和 swap-in 调度。模型接入层必须分开：token 语义、KV layout、写入接口、attention 消费方式都不同。

### 3.1 结论

GLM-5（及 DeepSeek-V3.2 等 NSA 模型）和 DeepSeek-V4 在 SGLang GPU 侧已经分成两套 HiSparse 实现，NPU 也应保持这个边界。

| 可共享 | 必须分开 |
|---|---|
| Host KV pool 管理、device hot buffer 分配、hit/miss/LRU、固定 shape hot-location 输出、swap-in 调度框架 | KV 压缩比、swap-in kernel layout、attention backend 接口、KV pool 结构 |

### 3.2 SGLang GPU 侧两条 HiSparse 路径对比

源码证据：

- 模型分类：`model_config.py` 的 `is_deepseek_nsa()` 包含 `GlmMoeDsaForCausalLM`、`DeepseekV3ForCausalLM`、`DeepseekV32ForCausalLM` 等；`is_deepseek_v4()` 只包含 `DeepseekV4ForCausalLM`。
- 验证分叉：`hisparse_hook.py:60-72`，V4 提前返回，跳过 NSA 的 dtype 检查。
- 协调器分叉：`hisparse_coordinator.py` 通过 `self.is_dsv4_hisparse` 区分两条路径。

| 维度 | NSA 路径（GLM-5, DeepSeek V3.2） | DSV4 路径（DeepSeek V4） |
|------|----------------------------------|--------------------------|
| 模型类 | `GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM` | `DeepseekV4ForCausalLM` |
| compress_ratio | 1（无压缩，1 token = 1 KV entry） | C4 pool = 4（4 token 压缩为 1 C4 KV entry；SWA/C128 另走各自分支） |
| Device pool | `HiSparseNSATokenToKVPool` | `HiSparseC4DevicePool`（仅 C4 pool 被替换） |
| Host pool | `MLATokenToKVPoolHost`（layout="layer_first"） | `DeepSeekV4SingleKVPoolHost` |
| swap-in kernel | `load_cache_to_device_buffer_mla`（device+host 都是 linear layout） | `load_cache_to_device_buffer_dsv4_mla`（page-padded device + linear host） |
| Indexer | `nsa_indexer.py`：FP8 paged MQA logits → `fast_topk_v2` | `dsv4/indexer.py`：`topk_transform_512` → raw indices |
| force_unfused_topk | HiSparse decode 时强制 True（返回 raw token positions） | 同上 |
| Attention 接入 | `nsa_backend.py` → `flashmla_kv` / `flashmla_sparse` / FA3 / tilelang | `deepseek_v4_backend.py` → FlashMLA with `extra_indices_in_kvcache` |
| FlashMLA 接口 | FP8 KV: `flash_mla_with_kvcache(..., indices=top_k_device_locs.unsqueeze(1))`；BF16 KV: `flash_mla_sparse_fwd(..., indices=top_k_device_locs.unsqueeze(1))` | `extra_indices_in_kvcache = top_k_device_locs`（作为额外稀疏部分） |
| KV offload 范围 | 全部 MLA KV | 仅 C4 KV（SWA/C128/indexer pool 不 offload） |
| item_size_bytes | `token_stride_size`（每个 token 的 KV 字节数） | `kv_cache_total_dim * dtype.itemsize` |

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
    → FP8 KV: flash_mla_with_kvcache(indices=page_table_1.unsqueeze(1))
    → BF16 KV: flash_mla_sparse_fwd(indices=page_table_1.unsqueeze(1))
    → 或 FA3 / tilelang 等其他 backend
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

DSV4 的 FlashMLA 调用比 NSA 多一层结构：SWA 部分用 `indices`，C4 sparse 部分用 `extra_indices_in_kvcache`（HiSparse 管理的 hot buffer）。

### 3.5 NPU 适配的分模型策略

| 顺序 | 理由 | 适配路径 |
|---|---|---|
| 先做 GLM-5/NSA | NPU 已有 `npu_lightning_indexer + npu_sparse_flash_attention`；`compress_ratio=1`；无压缩/解压；swap-in layout 更接近 linear；模型已在 NPU 上跑通推理 | 参考 GPU NSA HiSparse → 移植 residency 语义 → 对接 NPU SFA。方案 A=PA_BSND remap 主线，方案 B=BSND hot-slot 优化/备选，方案 C=改算子最后备选 |
| DeepSeek-V4 后做 | SGLang 有 `DeepseekV4ForCausalLM + DeepseekV4AttnBackend`（`deepseek_v4.py:1021`、`deepseek_v4_backend.py:331`），但 NPU 上 C4/C128/SWA 双路 attention 尚未接好；C4 `compress_ratio=4` 引入 `(last_locs - 3) // 4`；需要 page-padded layout；NPU SFA 没有 FlashMLA `extra_indices_in_kvcache` 等价接口；多 pool + 双路 attention 是独立工作 | 先完成 DSV4 基础推理（多 pool、C4 indexer、SWA+sparse 双路 attention）→ 再加 HiSparse（仅 C4 pool offload + hot buffer）→ attention 可能需要新 NPU kernel 或 SFA 的 DSV4 变体 |

### 3.6 公共组件复用

尽管接入层不同，以下组件可以设计为模型无关的公共层：

| 组件 | 接口 | NSA 使用 | DSV4 使用 |
|------|------|----------|-----------|
| Host KV pool 管理 | `alloc/free host slots` | `MLATokenToKVPoolHost` 等价 | `DeepSeekV4SingleKVPoolHost` 等价 |
| Device hot buffer 分配 | `alloc/free device buffer slots` | `HiSparseNSATokenToKVPool` 等价 | `HiSparseC4DevicePool` 等价 |
| LRU 状态机 | `hit/miss/evict/update` | per-layer, per-req | per-compress-layer, per-req |
| swap-in 调度 | `swap_in_selected_pages(top_k_result, layer_id) → device_locs` | linear layout | page-padded layout |
| Backup (D2H) | `backup_newest_token(layer_id)` | 每步 1 token | 每 compress_ratio 步 1 compressed item |
| 生命周期管理 | `staging → decode → finish` | compress_ratio=1 | C4 compress_ratio=4 |

公共抽象可以是一个 `HiSparseResidencyRuntime` 基类，参数化 `compress_ratio`、`item_size_bytes`、`layout_type`，由 NSA/DSV4 子类分别实现各自 layout 的 copy 和 attention 对接。

### 3.7 GLM-5 在 SGLang 中的特化点

GLM-5（`GlmMoeDsaForCausalLM`）直接继承 `DeepseekV2ForCausalLM`，几乎没有 attention 相关的模型特化。在 NSA HiSparse 路径里，唯一的 GLM-5 特化是：

- **Head padding**：`nsa_backend.py:364`——GLM-5 的 `num_q_heads` 可能小于 16（例如 64 heads / TP8 = 8），需要 pad to 16 才能对齐 FlashMLA tile 要求
- **Blackwell workaround**：`server_args.py:1796`——GLM-5 在 Blackwell GPU 上禁用 MHA_ONE_SHOT

除此之外，GLM-5 和 DeepSeek-V3.2 在 HiSparse 路径里完全一致。**NPU HiSparse 方案不需要为 GLM-5 单独设计**，只要做好通用 NSA/SFA 路径的适配即可覆盖所有 `is_deepseek_nsa` 模型。

### 3.8 为什么不能共用同一个 KV pool 实现

可以共用 HiSparse 驻留管理层（residency runtime：LRU / hit-miss / host pool / hot buffer / swap-in 调度），不能共用 KV pool。原因是 4 个接口契约不同：token 语义、内存 layout、写入接口、attention 消费方式。

#### 3.8.1 token 语义不同

GLM/NSA 的 HiSparse pool：一个原始 token → 一个 KV slot。`topk_indices` 选的就是原始序列 token id。

源码证据：`HiSparseNSATokenToKVPool.translate_loc_from_full_to_compressed()` 直接 `return full_indices`（`hisparse_memory_pool.py:90`），说明没有压缩 token 空间。

DSV4 C4 pool：4 个原始 token → 1 个 C4 compressed KV。只有 `(full_index + 1) % 4 == 0` 的位置才产生 C4 KV。

源码证据：`HiSparseC4DevicePool.translate_loc_from_full_to_compressed()`（`deepseek_v4_memory_pool.py:199-202`）：

```python
mask = (full_indices + 1) % self.compress_ratio == 0
compressed_indices = full_indices[mask] // self.compress_ratio
```

同一个 `loc=100`：GLM pool 里是第 100 个原始 token；DSV4 C4 pool 里要先判断是否合法，再映射到 compressed loc。

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

GLM/NSA：`top_k_device_locs` 直接作为 sparse attention 的主 `indices`。如果 decode backend 是 `flashmla_kv`（FP8 KV），实际调用是 `flash_mla_with_kvcache`：

```python
indices = page_table_1.unsqueeze(1)  # shape [B, 1, topk]
o, _ = flash_mla_with_kvcache(
    q=q_input,
    k_cache=kv_cache,
    indices=indices,
    block_table=torch.empty((B, 0), dtype=torch.int32, device=q.device),
    is_fp8_kvcache=True,
    ...)
# nsa_backend.py:1817-1836
```

如果 decode backend 是 `flashmla_sparse`（BF16 KV），实际调用是 `flash_mla_sparse_fwd`：

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

强行共用会错在四处：`loc` 语义错；copy stride 错（584 bytes packed uint8 vs `kv_cache_dim * dtype.itemsize`）；把 DSV4 packed uint8 当普通 BF16/FP8 tensor 读；NSA 主 `kv_cache` shape 和 DSV4 `extra_k_cache` shape 混用。

### 3.9 GLM Sparse Attention 与 DSV4 C4 的结构性差异（超越压缩比）

3.8 节是 KV pool 契约差异；本节是 attention 子系统差异。核心不是“DSV4 多了压缩比”，而是 GLM sparse attention 是这一层的主 attention，DSV4 C4 只是 SWA 之外的 extra 分支。

| 维度 | GLM/NSA Sparse Attention | DSV4 C4 Sparse Attention |
|------|--------------------------|--------------------------|
| attention 子系统 | `GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM`（`glm4_moe.py:1483`），走 `nsa_backend.py` 的 `flashmla_kv` / `flashmla_sparse` / FA3 / tilelang | 独立 `deepseek_v4_backend.py`；`compress_ratios[layer_id]` 决定 `0=SWA only`、`4=SWA+C4`、`128=SWA+C128`（`deepseek_v4.py:186-192`，`deepseek_v4_backend.py:972-979`） |
| attention 角色 | 该层主 attention：indexer top-k 后，FlashMLA 只读这些 KV | C4 是 extra 分支：只补 SWA 之外的远距离压缩信息；HiSparse 只涉及 `compress_ratio==4` 层 |
| indexer 候选空间 | 原始 token 序列，`config.index_topk` 通常 2048（`nsa_backend.py:226-239`），128K context 候选最多 128K | C4 compressed 序列：`c4_seq_lens=seq_lens//4`、`c4_page_size=page_size//4`（`metadata.py:47`），top-k=512 compressed tokens=2048 raw tokens；128K context 候选最多 32K |
| indexer 层分布 | 每层有 indexer slot（`get_num_indexer_layers = num_hidden_layers`，`model_config.py:148-149`），但可由 `index_topk_freq/index_topk_pattern` 做 `skip_topk` 复用（`deepseek_v2.py:1429-1447`） | indexer 层数为 `sum(1 for r in compress_ratios if r == 4)`（`model_config.py:150-152`）；只有 C4 层有 C4Indexer |
| HiSparse 接入点 | attention backend 内部：`nsa_backend.py:1612-1618` 调 `swap_in_selected_pages(req_pool_indices, seq_lens, topk_indices, layer_id)` | indexer 输出后、进入 backend 前：`dsv4/indexer.py:446-458` 调 `swap_in_selected_pages(..., compressed_seq_lens=c4_seq_lens, top_k_result=raw_indices, layer_id=compress_layer_id)` |
| indices 语义 | `page_table_1/top_k_device_locs` 是主 KV indices；FP8 进 `flash_mla_with_kvcache(indices=...)`，BF16 进 `flash_mla_sparse_fwd(indices=...)`，指向单一 `kv_cache` | `top_k_device_locs` 是 `extra_indices_in_kvcache`，指向独立 `extra_k_cache`；SWA 部分用另一组 `indices`，不走 HiSparse |
| HiSparse offload 范围 | 全部 MLA KV | 仅 C4 KV；SWA/C128/indexer 不 offload |
| seq_lens 给 swap-in | 原始 `seq_lens` | compressed `c4_seq_lens`（÷4） |

总结：GLM Sparse Attention 是“对原始 token 做 NSA top-k，然后主 attention 只读这些 KV”。DSV4 C4 是“先压缩生成 C4 KV，再在 SWA + extra sparse 复合 attention 中把 C4 当作 extra sparse KV 使用”（每层只有一种 extra 类型：C4 或 C128）。

所以二者的差异不只是 `compress_ratio`，还包括 6 个接口边界：attention 角色、indexer 作用域、层分布、top-k 空间、HiSparse 接入点、attention kernel 参数语义。

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


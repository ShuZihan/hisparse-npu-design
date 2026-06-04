# SGLang HiSparse 能力解读、架构分析与 NPU 适配方案

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md) | GPU HiSparse 能力解读 + 架构分析（数据结构、生命周期、swap-in kernel、stream 模型、路径对比、配置） |
| [02-npu-current-dsa.md](02-npu-current-dsa.md) | NPU 当前 DSA 实现现状、与 CUDA 差异、核心限制 |
| [03-npu-adaptation-plan.md](03-npu-adaptation-plan.md) | NPU HiSparse 适配方案分析（语义鸿沟、三条路线、Block POC、风险、分阶段落地、验收标准） |
| [04-nsa-vs-dsv4-differences.md](04-nsa-vs-dsv4-differences.md) | NSA（GLM-5/V3.2）与 DeepSeek-V4 的 HiSparse 路径差异（KV pool 契约、结构性差异、NPU 策略） |

---

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

落地分 6 个阶段（P0-P5），从 SFA 寻址探针 → Block hot buffer POC → Compact PA view → Direct physical SFA → Graph-first residency → 生产化。这些阶段是 POC 验证顺序，不是生产部署路径；详见 [03-npu-adaptation-plan.md § 4.6](03-npu-adaptation-plan.md#46-分阶段落地)。

**DeepSeek-V4 不应作为第一条 NPU HiSparse 主线。** 它依赖 DSV4 基础推理（多 pool、C4 indexer、SWA+C4 双路 attention）先在 NPU 上跑通，之后再单独加 C4 HiSparse。


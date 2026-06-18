# SGLang HiSparse 能力解读、架构分析与 NPU 适配方案

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md) | GPU HiSparse 能力解读 + 架构分析（数据结构、生命周期、swap-in kernel、stream 模型、路径对比、配置） |
| [02-npu-adaptation-plan.md](02-npu-adaptation-plan.md) | NPU HiSparse 适配方案（现状、主线方案、差异与对策、性能风险、分阶段落地） |
| [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md) | NSA（GLM-5/V3.2）与 DeepSeek-V4 的 HiSparse 路径差异（KV pool 契约、结构性差异、NPU 策略） |
| [ASSIGNMENTS.md](ASSIGNMENTS.md) | 三人分工（A: residency kernel / B: SFA 寻址 / C: 内存池+生命周期+同步），含 P0–P5 阶段—负责人对照 |

---

## 概述

**HiSparse** 解决长序列 decode 时 KV cache 全量常驻 GPU 显存太贵的问题。核心做法：全量 KV 放 CPU pinned memory，GPU 上只保留固定大小 hot buffer，每步按 indexer top-k 做 swap-in。GPU KV 显存从 O(seq_len) 降到 O(top_k)，128K 上下文省约 30 倍。核心语义是 **sparse KV residency**，不是 sparse attention。

**GPU 实现**三步完成：indexer 选 top-k → swap-in kernel 原子完成 hit/miss/LRU/copy → FlashMLA 按 physical hot slot 读 KV。详见 [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md)。

**NPU 适配**的目标架构对标 GPU HiSparse：让 AscendC kernel 在 kernel 内完成 hit/miss/LRU/copy。但 GPU 那条"kernel 内跨总线直读 host"在 NPU 上是**高风险实验路线（路径①）**——CANN `aclrtHostRegister` 映射地址官方注明"不能用于 memory copy"，统一地址路（`MallocHostWithCfg`+VA_FLAG）措辞相反但需上机验证 kernel 内 `DataCopy` 可用性（P0b 门控，§2.2.2）。**基准退路是路径②（staged DMA）**：kernel 只判定、搬运用既有 host 侧 DMA，必须先跑通。SFA 寻址首选**路线 A（BSND，不改算子）**——`sparse_indices` 填物理槽号；SFA 内核确有 BSND 一层寻址分支、公开 `torch_npu` 文档有 BSND+2048+图模式示例，但本地 CI 的 BSND 用例仅 `K=16/32`（`K=2048` 走 PA_BSND），故**生产形态 tuple 须经 P1 golden 验证**，非"CI 已实测"。详见 [02-npu-adaptation-plan.md](02-npu-adaptation-plan.md)。

**GLM-5 和 DeepSeek-V4 必须分开做**——先做 GLM-5/NSA 通用路径。详见 [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md)。

**落地顺序**：P0-P5 六阶段（P0a 构建链 → P0b host 直读探针 → P1 SFA 寻址探针（验证不改算子的路线 A/A′）→ P2 AscendC residency kernel → P3 寻址路线定稿（A/A′ 优先，不通过才退路线 C 改算子）→ P4 CANN Graph 验证 → P5 生产化），详见 [02 § 2.6](02-npu-adaptation-plan.md#26-分阶段落地)。

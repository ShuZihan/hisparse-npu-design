# SGLang HiSparse 能力解读、架构分析与 NPU 适配方案

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md) | GPU HiSparse 能力解读 + 架构分析（数据结构、生命周期、swap-in kernel、stream 模型、路径对比、配置） |
| [02-npu-adaptation-plan.md](02-npu-adaptation-plan.md) | NPU HiSparse 适配方案（现状、主线方案、差异与对策、性能风险、分阶段落地） |
| [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md) | NSA（GLM-5/V3.2）与 DeepSeek-V4 的 HiSparse 路径差异（KV pool 契约、结构性差异、NPU 策略） |

---

## 概述

**HiSparse** 解决长序列 decode 时 KV cache 全量常驻 GPU 显存太贵的问题。核心做法：全量 KV 放 CPU pinned memory，GPU 上只保留固定大小 hot buffer，每步按 indexer top-k 做 swap-in。GPU KV 显存从 O(seq_len) 降到 O(top_k)，128K 上下文省约 30 倍。核心语义是 **sparse KV residency**，不是 sparse attention。

**GPU 实现**三步完成：indexer 选 top-k → swap-in kernel 原子完成 hit/miss/LRU/copy → FlashMLA 按 physical hot slot 读 KV。详见 [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md)。

**NPU 适配**的主线方案是 1:1 对标 GPU 架构：通过 `aclrtHostRegister` 让 AscendC kernel 直接读取 Host KV（对标 CUDA PTX `ld.global.nc`），在 kernel 内完成 hit/miss/LRU/copy。NPU 特有工作是 SFA 寻址扩展（支持 physical hot slot 直接寻址）和 AscendC kernel 重写。详见 [02-npu-adaptation-plan.md](02-npu-adaptation-plan.md)。

**GLM-5 和 DeepSeek-V4 必须分开做**——先做 GLM-5/NSA 通用路径。详见 [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md)。

**落地顺序**：P0-P5 六阶段（aclrtHostRegister 探针 → SFA 寻址探针 → AscendC residency kernel → SFA 路线 C 原型 → CANN Graph 验证 → 生产化），详见 [02 § 2.6](02-npu-adaptation-plan.md#26-分阶段落地)。

# SGLang HiSparse 能力解读、架构分析与 NPU 适配方案

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md) | GPU HiSparse 能力解读 + 架构分析（数据结构、生命周期、swap-in kernel、stream 模型、路径对比、配置） |
| [02-npu-adaptation-plan.md](02-npu-adaptation-plan.md) | NPU HiSparse 适配方案（现状、主线方案、差异与对策、性能风险、分阶段落地） |
| [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md) | NSA（GLM-5/V3.2）与 DeepSeek-V4 的 HiSparse 路径差异（KV pool 契约、结构性差异、NPU 策略） |
| [04-npu-technical-paths.md](04-npu-technical-paths.md) | NPU HiSparse 技术路径说明（先讲 KV cache hierarchy，再讲 miss KV 怎么搬、SFA 怎么读 hot buffer、具体要开发哪些模块） |
| [ASSIGNMENTS.md](ASSIGNMENTS.md) | 三人分工（A: residency kernel / B: SFA 寻址 / C: 内存池+生命周期+同步），含 P0–P5 阶段—负责人对照 |

---

## 概述

**HiSparse** 解决长序列 decode 时 KV cache 全量常驻 GPU 显存太贵的问题。核心做法：全量 KV 放 CPU pinned memory，GPU 上只保留固定大小 hot buffer，每步按 indexer top-k 做 swap-in。GPU KV 显存从 O(seq_len) 降到 O(top_k)，128K 上下文省约 30 倍。核心语义是 **sparse KV residency**，不是 sparse attention。

**GPU 实现**三步完成：indexer 选 top-k → swap-in kernel 原子完成 hit/miss/LRU/copy → FlashMLA 按 physical hot slot 读 KV。详见 [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md)。

**NPU 适配**的目标架构对标 GPU HiSparse：让 AscendC kernel 完成 hit/miss/LRU，并尽量在 kernel 内把 miss KV 从 Host 搬进 NPU hot buffer。搬运有两种选择：首选 **kernel 自搬**，但要上机证明 kernel `DataCopy` 能读 registered/unified host memory；保底是 **Host 分段搬**，即 kernel 只判定 miss，Host 侧 H2D copy 后再补齐 SFA 坐标。SFA 读取当前应先保留 GLM-5 生产调用的 **PA_BSND + block_table**，通过重映射 `block_table` 让 SFA 读 hot buffer；**BSND hot-slot 直读**更像 GPU，但需要改 `layout_query/layout_kv` 和 query/rope/output 形态，作为优化/备选路线。两条路都对不齐 baseline 时才改 SFA 算子。详见 [04-npu-technical-paths.md](04-npu-technical-paths.md)。

**GLM-5 和 DeepSeek-V4 必须分开做**——先做 GLM-5/NSA 通用路径。详见 [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md)。

**实现顺序**：先跑通 NPU 自定义算子构建链路；再做 PA_BSND remap golden、BSND hot-slot golden 和 Host 直读探针；随后开发公共 runtime、residency judge kernel、搬运路径和 SFA adapter；最后做多层 decode、CANN Graph、CP/PD/异常回收等生产化验证。详见 [04-npu-technical-paths.md](04-npu-technical-paths.md)。

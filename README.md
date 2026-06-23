# SGLang HiSparse 与 NPU 适配文档

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-gpu-hisparse-architecture.md](01-gpu-hisparse-architecture.md) | GPU HiSparse：问题、资源模型、数据结构、生命周期、swap-in kernel、FlashMLA 对接、配置 |
| [02-npu-adaptation-plan.md](02-npu-adaptation-plan.md) | NPU 适配：现状差异、PA_BSND 主线、Host 搬运分叉、性能风险、阶段计划 |
| [03-nsa-vs-dsv4-differences.md](03-nsa-vs-dsv4-differences.md) | GLM-5/NSA 与 DeepSeek-V4：哪些驻留管理层可复用，哪些 KV/attention 契约必须分开 |
| [04-npu-technical-paths.md](04-npu-technical-paths.md) | NPU 技术路径短读版：token 9001 示例、A/B/C 寻址差异、开发顺序 |
| [ASSIGNMENTS.md](ASSIGNMENTS.md) | 三人分工：A residency kernel，B SFA 寻址，C pool/lifecycle/sync |

---

## 概述

**HiSparse** 解决长序列 decode 时 KV cache 全量常驻显存太贵的问题：全量 KV 放 CPU pinned memory，device 上只保留固定 hot buffer，每步按 indexer top-k swap-in。128K 上下文、`device_buffer_size=4096` 时，GPU KV 显存约省 30 倍。它的核心语义是 **sparse KV residency**，不是 sparse attention 本身。

**GPU** 路径：indexer 选 top-k → swap-in kernel 原子完成 hit/miss/LRU/copy → FlashMLA 按 physical device KV index 读 hot buffer。详见 [01](01-gpu-hisparse-architecture.md)。

**NPU** 目标对标 GPU，但不能照搬。搬运分两路：路径① kernel 内 `DataCopy` 直读 registered/unified host memory；路径② judge kernel 只产 miss list，Host staged H2D 后再补齐 SFA 坐标。寻址主线先保留 GLM-5 当前 **TND + PA_BSND + block_table**，用 remapped `block_table` 读 hot buffer；BSND hot-slot 是优化/备选；A/B 都不通才改 SFA 算子。详见 [02](02-npu-adaptation-plan.md) 和 [04](04-npu-technical-paths.md)。

**GLM-5/NSA 先做，DeepSeek-V4 后做**。公共驻留管理层可以共享，但 KV pool、压缩比、swap-in layout、attention 接入契约不同。详见 [03](03-nsa-vs-dsv4-differences.md)。

**实现顺序**：打通 `sgl-kernel-npu` 自定义算子链路 → P0b Host 直读探针与 judge-only kernel 并行 → 做 PA_BSND remap golden、BSND hot-slot golden → 开发驻留管理层、搬运路径、SFA 坐标转换 → 多层 decode、CANN Graph、CP/PD、异常回收验证。详见 [ASSIGNMENTS](ASSIGNMENTS.md)。

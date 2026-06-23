# NPU HiSparse 适配 —— 三人分工

> 配套阅读：整体规划见 [02 §2.6](02-npu-adaptation-plan.md)，分模型顺序见 [03 §3.5](03-nsa-vs-dsv4-differences.md)。本文把 P0–P5 切成三条可并行工作流。

## 切分思路

不按阶段横切（那样三人前期全堵在 P0、后期互相等），而是按**三大难点**纵切成三条流：

| 负责人 | 工作流 | 对应难点 | 覆盖章节 / 阶段 |
|--------|--------|----------|-----------------|
| **A** | Residency Kernel | ①谁判定 hit/miss 并搬 KV | §2.3.1、§2.7；**P0 → P2** |
| **B** | SFA 寻址 | ②hot location 怎么变成 SFA 坐标 | §2.3.2；**P1 → P3** |
| **C** | 内存池 + 生命周期 + 同步 | ③搬运的骨架 | §2.3.3、§2.4、§2.8 |

---

## A —— Residency Kernel（难点①：谁判定 hit/miss 并搬 KV）

**职责**：写 NPU residency kernel，判断 hit/miss、更新 LRU，并在路径①里把 miss KV 从 Host 搬进 hot buffer。它不在本仓库实现：所有 NPU 自定义算子来自外部包 `sgl-kernel-npu`（tag `2026.05.01` 已拉到 `../sgl-kernel-npu/`，见 [02 §2.2.1](02-npu-adaptation-plan.md)）。A 的第一步是 P0a：打通 `build.sh`、`python/` 注册和 `tests/` 单测。`csrc/cache_location_assign/` 已示范 `GetBlockIdx` tiling、`TPipe/TBuf/TQue`、`DataCopy/DataCopyPad`、`SetFlag/WaitFlag<HardEvent>`，可作模板。

**每步 decode 做什么**（对应 §2.7.3 三个 Phase）：

1. **Phase 1 判断（Vector 主导）**：indexer 给出本层 top-k token 清单。用 **Vector 单元批量比对**哪些 token 已在 hot buffer（hit）、哪些不在（miss）。**必须用 Vector**，不能用 Scalar 逐个比；NPU 的 Scalar 单元只适合地址和调度（§2.7.5）。
2. **Phase 2 搬运（Scalar 派发 + MTE 流水线）**：对每个 miss，Scalar 计算 Host KV 源地址和 hot buffer 目标位置，交 **MTE（DMA）** 搬。DataCopy **不支持一次给一串地址随机抓**，只能逐块发，用 **double buffer** 让当前块搬运和下一块地址计算重叠。
3. **Phase 3 输出**：产出 `hot_locations`。方案 A 是 hot block/page 映射；方案 B 是 hot slot list。

**阶段任务**：

| 阶段 | 干什么 |
|------|--------|
| **P0a（前置，越早越好）** | 在已拉取的 `../sgl-kernel-npu/` 跑通 `build.sh`，以 `csrc/helloworld/`（hello-world 模板）新增一个自定义 AscendC 算子并经 `python/` 注册 `torch.ops.npu.*`、`tests/` 加单测——本仓库无 AscendC 源码，不先搭好这条链路，P2 的 kernel 没处写、没法编译 |
| **P0b（第一周冲刺）** | 最小探针 kernel，验证 §2.2 三个未知数：`aclrtHostRegister` 注册的 host 地址能否在 kernel 里 DataCopy 直读？延迟多大？能否被 Graph capture？它决定搬运放在 kernel 内还是 Host staged DMA——见下方风险 |
| **P2** | P0a 打通后写 residency kernel，对齐 C 的 Python 参考实现。P0b 通过则包含上面三个 Phase；P0b 不通过则先做 judge-only，copy 交给 C 的 staged DMA |

**最大风险**：P0b 决定搬运是否并进 kernel。判定平面（排序求交、GatherMask、归约 LRU）走已验证 SIMD API，不依赖 host 直读，所以 judge-only kernel 可先写。`device 读写 registered host` 能力有 A3/A2 官方背书，但 HiSparse 需要的 kernel 内 `DataCopy` 直读 host 未明说；`HostGetDevicePointer` 路和 `MallocHostWithCfg`+VA_FLAG 统一地址路的官方措辞相反（[02 §2.2.2](02-npu-adaptation-plan.md)）。

- **路径①（P0b 通过、带宽够）**：judge + copy + 输出在**一个 kernel 内**完成，也就是上面三个 Phase。
- **路径②（P0b 不通过 / 带宽不够）**：搬运退出 kernel。A 的 kernel 退化为 **judge-only**（只判定、只产 miss_list / evict_slots / hit 部分的槽号），实际搬运改由 **C 在 stream 上发显式 H2D staged DMA**。注意搬运 API 要按 hot buffer layout 选：既有 `transfer_kv_dim_exchange` 更适合 PA/page 布局（方案 A），**BSND/token 布局（方案 B）需新写 staged copy**（[02 §2.2 路径 ② 架构](02-npu-adaptation-plan.md)）。每步 decode 多一趟 device→host→device 往返，Graph 需分段。

**起跑策略**：judge-only kernel 立即开写；P0b 并行探，决定是否把搬运并进 kernel。

---

## B —— SFA 寻址（难点②：hot location 怎么变成 SFA 坐标）

**职责**：让 `npu_sparse_flash_attention` 正确读到 hot buffer KV。B 消费 A 输出的 `hot_locations`，产出 SFA 需要的 `sparse_indices` / `block_table` / layout 参数。

**三条方案**（细节和 token `9001` 例子见 [04 §4](04-npu-technical-paths.md)，源码证据见 [02 §2.3.2](02-npu-adaptation-plan.md)）：

| 方案 | B 的交付 | 验证重点 |
|------|----------|----------|
| A：PA_BSND remap | 保持 `layout_query="TND"`、`layout_kv="PA_BSND"`，重映射 `block_table` 指向 hot buffer block | remapped `block_table` 是否能读 hot buffer；page 放大后 hot buffer/H2D 成本是否可接受 |
| B：BSND hot-slot | hot buffer 作为 BSND KV，`sparse_indices` 填 hot slot，`block_table=None` | query/rope/output reshape、`s2StartIdx`、`actual_seq_lengths_kv`、K=2048 生产形态 golden |
| C：改 SFA | 仅当 A/B 不可行时，新增 hot-slot 调用契约 | op host、tiling、kernel、测试、分发版本都要覆盖 |

**阶段任务**：

| 阶段 | 干什么 |
|------|--------|
| **P1** | SFA 寻址探针：先验证方案 A（PA_BSND remapped `block_table`）能否正确读 hot buffer，再验证方案 B（BSND 小 hot buffer + `sparse_indices` 直填 hot slot），做单层 golden test。这是**接口合法性验证**，不是改算子 |
| **P3** | P1 通过则用 A/B 做多层 decode 对齐；不通过才退方案 C 改算子内部寻址 |

**特点**：B 可从第一天独立开干，不等 A 的 P0 结论；P1 只验证调用契约是否合法，不改算子。

> 注意：`npu_sparse_flash_attention` 的源码**不在 `sgl-kernel-npu`**，但**在另一仓库 `ops-transformer/attention/sparse_flash_attention`、已可读**。方案 A/B 不需要改源码，只在 Python 侧调对 `layout_kv`/`sparse_indices`/`block_table` 即可；只有方案 C 才需触及算子内部并自带构建分发。`lightning_indexer`（top_k 引擎）的源码在 `sgl-kernel-npu` 包里（`csrc/lightning_indexer/op_kernel`，向量排序已有代码依据）。

---

## C —— 内存池 + 生命周期 + 同步（难点③：搬运的骨架）

**职责**：搭 Host pool、hot buffer、allocator，以及 staging/decode/finish 生命周期和 stream/event 同步。依赖最少，可立刻开干。

> 注意：C 要建的是 **两个 pool + 一个 allocator**（§2.3.3），不是三个 pool。

| 组件 | 类型 | 职责 | 对标 GPU |
|------|------|------|----------|
| `NPUMLATokenToKVPoolHost` | Host pool | 存全量 KV，`aclrtHostRegister` 注册后供 kernel 或 staged DMA 读取 | `MLATokenToKVPoolHost` |
| `NPUHiSparseMLATokenToKVPool` | Device pool | 小 hot buffer，**布局随寻址方案定**：方案 A = PA_BSND（`[hot_blocks, block_size, N, D]` + 重映射 block_table）；方案 B = BSND（`[B, device_buffer_size, N, D]`，`sparse_indices` 直填 hot slot）。只分配 `device_buffer_size`（如 4k）槽 | `HiSparseNSATokenToKVPool` |
| `NPUHiSparseAllocator` | Allocator（调度器） | logical↔device 映射 + staging/decode/finish 生命周期 | `HiSparseTokenToKVPoolAllocator` |

**三者关系**：两个 pool 存 KV，allocator 维护 logical↔device/host 映射，调度 swap-in、backup、新 token reserved slot。

**不归 C 管、但常被问到的第四块**（§2.3.4）：Indexer KV 体积小（约 2GB），**全量常驻 NPU、不 offload**，现有代码不动。它不参与 HiSparse 搬运。

**阶段任务**（无 P 编号门控，但 P2 前要交付参考实现）：

| 工作 | 干什么 |
|------|--------|
| pool/allocator | 上表两个 pool + 一个 allocator |
| 生命周期（§2.4） | staging（全量 KV 备份到 Host、释放全量 device slots、分配 hot buffer）→ decode（每步分 slot、记账、增量备份）→ finish（清状态并归还 slots）；含 PD 混合/分离、长短序列分支 |
| 同步（§2.8） | `main_stream` / `backup_stream` 双流 + event；守住“Host 副本写完前，hot buffer 对应位置不能被覆写” |
| **Python 参考实现** | 给 A 的 P2 做对齐 golden，让 A 写 kernel 时有逐步对照 |

**P0 风险波及**：若 P0b 走路径②，C 要负责显式 H2D staged DMA 和 stream 同步，Graph 可能分段。`transfer_kv_dim_exchange` 是纯 host API：索引 `.cpu()`，CPU 逐页 `for`，每页 `aclrtMemcpy2dAsync`，一次搬该页全部 layer，要求 `device_k.dim()==5`，更适合方案 A 的 PA/page hot buffer；方案 B 的 BSND/token hot buffer 需新写 staged copy。

---

## 协作约定

1. **交汇数据结构**：A 的输出 = B 的输入，统一叫 `hot_locations`。方案 A 至少包含 hot page/block 映射；方案 B 是 hot slot list。第一周先把 shape 和语义敲成契约。
2. **起跑姿势**（依赖关系决定）：
   - **A 第一周冲 P0a + judge-only kernel，P0b 并行探**（P0a 打通 `sgl-kernel-npu` 构建链路是写任何 kernel 的前提，可拿 `csrc/cache_location_assign/` 当模板；判定平面不依赖 host 直读，judge-only kernel 可立即开写；P0b 探针并行做，决定要不要把搬运并进 kernel，它不再卡住全局，见 [02 §2.7.5 架构判断](02-npu-adaptation-plan.md)）；
   - **B、C 不必等 A，立刻并行开干**（B 探 SFA 寻址、与 A 共享 `sgl-kernel-npu` 通路；C 搭 pool + 生命周期 + Python 参考实现 + 摸熟 `transfer_kv_dim_exchange` staged DMA 通路）。
3. **集成节点**：
   - **P4**：A 的 kernel + B 的 SFA + C 的同步端到端 CANN Graph capture/replay，三条工作流首次集成调试；
   - **P5**：多请求、异常回收、CP 路径（§2.3.2 标注的未覆盖点）按子任务再分。

## 阶段—负责人对照（P0–P5）

| 阶段 | 主负责 | 协作 |
|------|--------|------|
| P0a `sgl-kernel-npu` 构建链路 | **A** | B 共享通路 |
| P0b aclrtHostRegister 探针（搬运分叉） | **A** | C 留意结论（决定路径①/②） |
| P1 SFA 寻址探针 | **B** | — |
| P2 AscendC residency kernel（路径②为 judge-only + C 的 staged DMA） | **A** | C 提供 Python 参考实现；路径②下 C 接搬运 |
| P3 SFA 寻址定稿（先方案 A，再方案 B，均不改算子） | **B** | A 提供 `hot_locations` |
| P4 CANN Graph 验证 | **A + B + C** | 三条工作流集成 |
| P5 生产化（含 CP 路径） | **A + B + C** | 按子任务分 |

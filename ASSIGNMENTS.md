# NPU HiSparse 适配 —— 三人分工

> 配套阅读：整体规划见 [02 §2.6 分阶段落地](02-npu-adaptation-plan.md)，分模型顺序见 [03 §3.5](03-nsa-vs-dsv4-differences.md)。
> 本文把 P0–P5 阶段按**三条相对独立的工作流**切给三个人，让前期能真正并行、不互相堵。

## 切分思路

不按阶段横切（那样三人前期全堵在 P0、后期互相等），而是按**三大难点**纵切成三条流：

| 负责人 | 工作流 | 对应难点 | 覆盖章节 / 阶段 |
|--------|--------|----------|-----------------|
| **A** | Residency Kernel | ①谁来搬页 | §2.3.1、§2.7；**P0 → P2** |
| **B** | SFA 寻址 | ②页码怎么对 | §2.3.2；**P1 → P3** |
| **C** | 内存池 + 生命周期 + 同步 | ③搬运的骨架 | §2.3.3、§2.4、§2.8 |

（比方承接 01/02：显存=小书桌，host=大仓库，按需搬页=swap-in，淘汰最久未看=LRU。）

---

## A —— Residency Kernel（难点①：谁来搬页）

**一句话职责**：写在 NPU 上"判断哪页在桌上、把缺的页从仓库搬上来、挤掉最久没看的页"的 AscendC kernel。**技术风险最高、最核心**——GPU 用一个 kernel 搞定的事，NPU 硬件不支持照抄，而且**这个 kernel 的源码不在本仓库**：本仓库 `sglang-v0.5.12` 没有任何 AscendC 算子源码，所有 NPU 自定义算子都来自外部包 `sgl-kernel-npu`（已按 sglang CI 钉定的 tag `2026.05.01` 拉取到 `../sgl-kernel-npu/`，[02 §2.2.1](02-npu-adaptation-plan.md)）。所以 A 的第一件事不是写 kernel，而是**打通 `sgl-kernel-npu` 的本地构建/贡献链路**（P0a）。**好消息**：包里 `csrc/cache_location_assign/` 已经是一个语义最接近驻留逻辑的现成 AscendC 算子,示范了 `GetBlockIdx` tiling / `TPipe`+`TBuf`/`TQue` 管 UB / `DataCopy`+`DataCopyPad` 走 MTE / `SetFlag/WaitFlag<HardEvent>` 卡流水的全套骨架——A 的新 kernel 可直接以它为模板(§2.2.1 (5))。

**每步 decode 做什么**（对应 §2.7.3 三个 Phase）：

1. **Phase 1 判断（Vector 主导）**：indexer 给来"这层要看的页码清单"（top_k tokens）。用 **Vector 单元批量比对**哪些已在书桌（hit）、哪些不在（miss）。**必须用 Vector**，不能用 Scalar 逐个比——NPU 的 Scalar 单元很弱（§2.7.5）。
2. **Phase 2 搬运（Scalar 派发 + MTE 流水线）**：对每个 miss，Scalar 算"从仓库哪读、写书桌哪槽"，交 **MTE（DMA）** 搬。DataCopy **不支持一次给一串地址随机抓**，只能逐页发，用 **double buffer** 让"搬这页"和"算下页地址"重叠掩盖延迟。
3. **Phase 3 输出**：产出 `top_k_device_locs`——每页在书桌上的物理槽号。这就是交给 B 的清单。

**阶段任务**：

| 阶段 | 干什么 |
|------|--------|
| **P0a（前置，越早越好）** | 在已拉取的 `../sgl-kernel-npu/` 跑通 `build.sh`，以 `csrc/helloworld/`（hello-world 模板）新增一个自定义 AscendC 算子并经 `python/` 注册 `torch.ops.npu.*`、`tests/` 加单测——本仓库无 AscendC 源码，不先搭好这条链路，P2 的 kernel 没处写、没法编译 |
| **P0b（门控，第一周冲刺）** | 最小探针 kernel，验证 §2.2 三个未知数：`aclrtHostRegister` 注册的 host 地址能否在 kernel 里 DataCopy 直读？延迟多大？能否被 Graph capture？**这是二元分叉点**——见下方风险 |
| **P2** | P0 通过后写完整 residency kernel（上面三个 Phase），对齐 C 的 Python 参考实现 |

**最大风险（P0b 决定走哪条路，但不再是"全盘皆输"的赌注）**：P0b 的结论决定架构形态,但**两条路都有 proven 退路**——判定平面(排序求交 hit 判定 + GatherMask 收集 + 归约 LRU)全部走 device 上已验证的 SIMD API、不依赖 host 直读([02 §2.7.3.1](02-npu-adaptation-plan.md)),所以 **A 不必等 P0b 结论才能动手写判定 kernel**。而且"device 读写 registered host"本身是昇腾官方明文支持的能力(A3/A2 √,[02 §2.2.2](02-npu-adaptation-plan.md)),P0b 只是实测"kernel 内 DataCopy 这一具体用法 + EP 形态跨总线带宽"——风险等级比原先估计的低。

- **路径①（P0b 通过、带宽够）**：1:1 对标 GPU，judge + copy + 输出在**一个 kernel 内**一气呵成，就是上面三个 Phase。
- **路径②（P0b 不通过 / 带宽不够）**：搬运退出 kernel。A 的 kernel 退化为 **judge-only**（只判定、只产 miss_list / evict_slots / hit 部分的槽号），实际搬运改由 **C 在 stream 上发显式 H2D staged DMA**（复用仓库既有的 `transfer_kv_dim_exchange`，[02 §2.2 备选架构](02-npu-adaptation-plan.md)）。每步 decode 多一趟 device→host→device 往返，Graph 需分段。

**所以 A 的起跑策略调整**:judge-only kernel(路径②也需要、且不依赖 P0b)可**立即开写**;P0b 探针并行做、用来决定要不要把搬运也并进 kernel(路径①优化),而不是卡在它前面干等。

---

## B —— SFA 寻址（难点②：页码怎么对）

**一句话职责**：让 NPU sparse attention 算子（`npu_sparse_flash_attention`）**正确读到 A 输出的物理槽号对应的 KV**。**首选不改算子**——SFA 内核源码 + CI 实测证实它已有现成机制能吃 hot slot。

**关键修正（内核源码 + CI 推翻了"必须改算子"的旧前提）**：读 SFA 真源码（`ops-transformer/attention/sparse_flash_attention`）+ CI paramset 后确认：
- `sparse_indices` 是"离散取 kvCache 的索引"清单（与 GPU `indices` 同性质），但 **kernel 内 `s2Idx = sparse_indices + s2StartIdx`，索引是窗口相对、非绝对槽号**（`service_vector_mla.h:207`，`util_regbase.h:70`）；
- 两种寻址由编译期 `isPa` 选：PA_BSND 查 block_table（两级），**BSND 走 `realkeyOffset = boIdx*s2Size + s2Idx`（一级直址，不绑 block_table，`:233`）**；
- `bsnd_basic`/`pa_bsnd` 等都是 `ENABLED_PARAMS` 里的 CI 实测路径，**全 `sparse_block_size=1`**——散读由 SFA 自己扛（内部做"指令缩减 + 搬运聚合"），不是 B 要解决的问题；
- 粒度约束：arch35（950 类）硬编 `sparse_block_size=1`（仅 token-wise），block-wise 仅 arch22；
- 支持图模式（`return_softmax_lse=False` 时）。

**三条路线（按改动深度，[02 §2.3.2](02-npu-adaptation-plan.md)）**：

- **路线 A（首选·不改算子）**：hot buffer 走 **BSND** 当 key/value，`sparse_indices` 填 hot slot 号。对位 GPU"扁平池 + 物理槽清单"，token 粒度，无需碰算子源码。**两个约束**：① BSND 要求 `layout_query==layout_kv`，故须把 query 从现网 TND 一并切到 BSND（query 侧 reshape）；② 槽号非绝对填入，须处理 `s2StartIdx` 基址（确认 decode 下为 0 或喂入前扣除）。
- **路线 A′（不改算子·页粒度）**：保持 PA_BSND，重映射 `block_table` 指向 hot buffer。代价是页粒度驻留，但 query 可继续用 TND（PA_BSND 允许 query/kv 布局不一致），改动比 A 更局部。`sparse_block_size` 仍为 1，不受 arch35 限制。
- **路线 C（兜底·改算子）**：仅当 A/A′ 经 P1 验证不可行时，才给 SFA 加 `PHYSICAL_HOT_SLOT` 模式。SFA 源码可读（在 `ops-transformer`，不在 `sgl-kernel-npu`），但改它需自带算子构建分发 + 确认生产二进制版本对齐，改动最深最被动。

**阶段任务**：

| 阶段 | 干什么 |
|------|--------|
| **P1** | SFA 寻址探针：先验证路线 A（BSND 小 hot buffer + sparse_indices 直填 hot slot）或 A′（重映射 block_table）能否正确读到，做单层 golden test。这是**接口合法性验证**，不是改算子 |
| **P3** | P1 通过则用 A/A′ 做多层 decode 对齐；不通过才退路线 C 改算子内部寻址 |

**特点**：与 A、C 高度解耦——B 只关心"给我一张物理槽号清单，我让 SFA 正确读到对应 KV"。**B 可从第一天独立开干**，不必等 A 的 P0 结论。**且因首选不改算子，B 的工作量和风险比原先估计的低很多**。

> 注意：`npu_sparse_flash_attention` 的源码**不在 `sgl-kernel-npu`**，但**在另一仓库 `ops-transformer/attention/sparse_flash_attention`、已可读**。路线 A/A′ 不需要改源码——只在 Python 侧调对 `layout_kv`/`sparse_indices`/`block_table` 即可；只有兜底路线 C 才需触及算子内部并自带构建分发（最被动）。`lightning_indexer`（top_k 引擎）的源码在 `sgl-kernel-npu` 包里（`csrc/lightning_indexer/op_kernel`，向量排序实证）。

---

## C —— 内存池 + 生命周期 + 同步（难点③：搬运的骨架）

**一句话职责**：搭起"仓库、书桌、和指挥搬运的管理员"，以及四阶段生命周期和多流水线同步。**最实、依赖最少，可立刻开干**。

> 注意：C 要建的是 **两个 pool + 一个 allocator**（§2.3.3），不是三个 pool。

| 组件 | 类型 | 比方 | 职责 | 对标 GPU |
|------|------|------|------|----------|
| `NPUMLATokenToKVPoolHost` | Host pool | 仓库 | 存全量 KV，`aclrtHostRegister` 注册供 kernel 直读 | `MLATokenToKVPoolHost` |
| `NPUHiSparseMLATokenToKVPool` | Device pool | 书桌 | 小 hot buffer，PA_BSND layout，只分配 `device_buffer_size`（如 4k）槽 | `HiSparseNSATokenToKVPool` |
| `NPUHiSparseAllocator` | Allocator（调度器） | 管理员 | logical↔device 映射 + staging/decode/finish 生命周期 | `HiSparseTokenToKVPoolAllocator` |

**三者关系**：管理员（allocator）拿着账本，在仓库（host pool）和书桌（device pool）之间调度——哪页从仓库搬上桌、哪页写回仓库、新 token 放书桌哪槽。两个 pool 是"存东西的地方"，allocator 是"指挥搬运的人"。

**不归 C 管、但常被问到的第四块**（§2.3.4）：Indexer KV 体积小（~2GB），**全量常驻 NPU、不下沉仓库**，现有代码不动。它不参与 HiSparse 搬运。

**阶段任务**（无 P 编号门控，但 P2 前要交付参考实现）：

| 工作 | 干什么 |
|------|--------|
| pool/allocator | 上表两个 pool + 一个 allocator |
| 生命周期（§2.4） | staging（整本归档进仓库、回收大显存、分配小书桌）→ decode（每步分槽、记账、增量备份）→ finish（清账还槽）；含 PD 混合/分离、长短序列分支 |
| 同步（§2.8） | `main_stream` / `backup_stream` 双流 + event 红绿灯；守住"仓库没存好副本前别覆写书桌" |
| **Python 参考实现** | 给 A 的 P2 做对齐 golden——**关键润滑剂**，让 A 写 kernel 时有对照物 |

**P0 风险波及（C 要为两条路径都做准备）**：A 的 P0b 是二元分叉。**若走路径②**（kernel 读不了 host），swap-in 的实际搬运落到 C 头上——C 要在 swap-in 路径用既有的 `transfer_kv_dim_exchange` / `aclrtMemcpyAsync` 发显式 H2D staged DMA，并加 stream 同步，会打断 Graph。读过 `transfer_kv_dim_exchange` 的真实实现（[..../transfer_kv_dim_exchange.cpp](../sgl-kernel-npu/csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp)）后这条通路的形态很清楚:它是**纯 host_api**——索引先 `.cpu()`、CPU 上逐页 `for` 循环、每页一发 `aclrtMemcpy2dAsync`(一次搬该页全部 layer，靠 5D 布局的 layer/page 维互换)。C 可直接复用或仿照它，无需自己写 kernel。所以这条 staged DMA 通路无论哪条路径都值得早点摸熟——路径②它就是主力搬运手段。**密切留意 A 的 P0b 结论。**

---

## 协作约定

1. **交汇就一个数据结构**：A 的输出 = B 的输入 = `top_k_device_locs`（物理槽号清单）。**第一周三人先把它的 shape 和语义敲成契约**，之后各写各的，P3 自然能接上。
2. **起跑姿势**（依赖关系决定）：
   - **A 第一周冲 P0a + judge-only kernel,P0b 并行探**（P0a 打通 `sgl-kernel-npu` 构建链路是写任何 kernel 的前提,可拿 `csrc/cache_location_assign/` 当模板;判定平面不依赖 host 直读,judge-only kernel 可立即开写;P0b 探针并行做,决定要不要把搬运并进 kernel——它不再卡住全局,[02 §2.7.5 架构判断](02-npu-adaptation-plan.md)）；
   - **B、C 不必等 A，立刻并行开干**（B 探 SFA 寻址、与 A 共享 `sgl-kernel-npu` 通路；C 搭 pool + 生命周期 + Python 参考实现 + 摸熟 `transfer_kv_dim_exchange` staged DMA 通路）。
3. **合体节点**：
   - **P4**：A 的 kernel + B 的 SFA + C 的同步端到端 CANN Graph capture/replay，三流第一次合体一起调；
   - **P5**：多请求、异常回收、CP 路径（§2.3.2 标注的未覆盖点）按子任务再分。

## 阶段—负责人对照（P0–P5）

| 阶段 | 主负责 | 协作 |
|------|--------|------|
| P0a `sgl-kernel-npu` 构建链路 | **A** | B 共享通路 |
| P0b aclrtHostRegister 探针（门控分叉） | **A** | C 留意结论（决定路径①/②） |
| P1 SFA 寻址探针 | **B** | — |
| P2 AscendC residency kernel（路径②为 judge-only + C 的 staged DMA） | **A** | C 提供 Python 参考实现；路径②下 C 接搬运 |
| P3 SFA 寻址定稿（首选路线 A，不改算子） | **B** | A 提供 `top_k_device_locs` |
| P4 CANN Graph 验证 | **A + B + C** | 三流合体 |
| P5 生产化（含 CP 路径） | **A + B + C** | 按子任务分 |

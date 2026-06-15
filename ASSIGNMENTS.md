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

贯穿全文的比方（承接 01/02 文档）：GPU/NPU 显存是**小书桌**，host 内存是**大仓库**，按需把页搬上桌 = swap-in，放回最久没看的页 = LRU。

---

## A —— Residency Kernel（难点①：谁来搬页）

**一句话职责**：写在 NPU 上"判断哪页在桌上、把缺的页从仓库搬上来、挤掉最久没看的页"的 AscendC kernel。**技术风险最高、最核心**——GPU 用一个 kernel 搞定的事，NPU 硬件不支持照抄，而且**这个 kernel 的源码不在本仓库**：本仓库 `sglang-v0.5.12` 没有任何 AscendC 算子源码，所有 NPU 自定义算子都来自外部包 `sgl_kernel_npu`（[02 §2.2.1](02-npu-adaptation-plan.md)）。所以 A 的第一件事不是写 kernel，而是**打通 `sgl_kernel_npu` 的本地构建/贡献链路**（P0a）。

**每步 decode 做什么**（对应 §2.7.3 三个 Phase）：

1. **Phase 1 判断（Vector 主导）**：indexer 给来"这层要看的页码清单"（top_k tokens）。用 **Vector 单元批量比对**哪些已在书桌（hit）、哪些不在（miss）。**必须用 Vector**，不能用 Scalar 逐个比——NPU 的 Scalar 单元很弱（§2.7.5）。
2. **Phase 2 搬运（Scalar 派发 + MTE 流水线）**：对每个 miss，Scalar 算"从仓库哪读、写书桌哪槽"，交 **MTE（DMA）** 搬。DataCopy **不支持一次给一串地址随机抓**，只能逐页发，用 **double buffer** 让"搬这页"和"算下页地址"重叠掩盖延迟。
3. **Phase 3 输出**：产出 `top_k_device_locs`——每页在书桌上的物理槽号。这就是交给 B 的清单。

**阶段任务**：

| 阶段 | 干什么 |
|------|--------|
| **P0a（前置，越早越好）** | 打通外部包 `sgl_kernel_npu` 的本地构建/编译/加载——本仓库无 AscendC 源码，不先搭好这条链路，P2 的 kernel 没处写、没法编译。先能跑通一个 hello-world AscendC 算子即可 |
| **P0b（门控，第一周冲刺）** | 最小探针 kernel，验证 §2.2 三个未知数：`aclrtHostRegister` 注册的 host 地址能否在 kernel 里 DataCopy 直读？延迟多大？能否被 Graph capture？**这是二元分叉点**——见下方风险 |
| **P2** | P0 通过后写完整 residency kernel（上面三个 Phase），对齐 C 的 Python 参考实现 |

**最大风险（P0b 是二元分叉，不是单纯调优）**：P0b 的结论决定整个架构走哪条路，A 和 C 后面怎么写全看这一条——

- **路径①（P0b 通过，kernel 能直读 host）**：1:1 对标 GPU，judge + copy + 输出在**一个 kernel 内**一气呵成，就是上面三个 Phase。
- **路径②（P0b 不通过，kernel 读不了 host）**：搬运退出 kernel。A 的 kernel 退化为 **judge-only**（只判定、只产 miss_list / evict_slots / hit 部分的槽号），实际搬运改由 **C 在 stream 上发显式 H2D staged DMA**（复用仓库既有的 `transfer_kv_dim_exchange`，[02 §2.2 备选架构](02-npu-adaptation-plan.md)）。每步 decode 多一趟 device→host→device 往返，打断 Graph。

**A 第一周的 P0b 结论直接决定 A 和 C 后面怎么写**，所以必须最先冲。

---

## B —— SFA 寻址（难点②：页码怎么对）

**一句话职责**：改 NPU sparse attention 算子（`npu_sparse_flash_attention`），让它**直接吃 A 输出的物理槽号**，而不是非要自己查 block_table 翻译一遍。

**为什么必须改**（NPU 比 GPU 多出来的活）：

- A 输出的 `top_k_device_locs` 已是**现成的格子号**（书桌物理槽）。
- 但 NPU SFA 默认走**两级寻址**：把输入当 logical 订单号，内部查 block_table 翻成格子号。查出来的是**旧仓库的格子，不是新书桌的格子**——直接用会读错。
- 所以加一条**新路**："清单上已经是格子号了，别查表，直接 `hot_buffer_ptr + indices[i] * stride` 取"。

**两条技术路线**：

- **路线 C（推荐）**：给 SFA 新增 `PHYSICAL_HOT_SLOT` 寻址模式，跳过 block_table。改动集中在算子内部 `DataCopyPA` / `GetKeyGmOffset` 分支。**改算子内部一处，三个调用点（§2.3.2 表里的 :949 / :739 / :761）全受益**，不用改三遍 Python。
- **路线 B（备选）**：若路线 C 的"散乱地址逐个读"性能不达标（§2.5 风险），退而把 top-k 页先 gather 成连续 buffer 让 SFA 连续读。代价是牺牲 LRU 缓存收益，保住连续读性能。

**阶段任务**：

| 阶段 | 干什么 |
|------|--------|
| **P1** | SFA 寻址探针：读懂算子源码，精确定位 `DataCopyPA` / `GetKeyGmOffset` 改点，做单层 golden test |
| **P3** | 实现路线 C 原型，扩展寻址 ABI，多层 decode 对齐 full-resident；C 性能不行则切路线 B |

**特点**：与 A、C 高度解耦——B 只关心"给我一张物理槽号清单，我让 SFA 正确读到对应 KV"。**B 可从第一天独立开干**，不必等 A 的 P0 结论。

> 注意：`npu_sparse_flash_attention` 是 `torch_npu` / `sgl_kernel_npu` 的预编译算子，**源码不在本仓库**（[02 §2.2.1](02-npu-adaptation-plan.md)）。B 的 P1 第一步就是定位并拿到该算子源码、打通其构建链路（与 A 的 P0a 是同一条 `sgl_kernel_npu` 贡献通路，可共享成果）。

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

**P0 风险波及（C 要为两条路径都做准备）**：A 的 P0b 是二元分叉。**若走路径②**（kernel 读不了 host），swap-in 的实际搬运落到 C 头上——C 要在 swap-in 路径用既有的 `transfer_kv_dim_exchange` / `aclrtMemcpyAsync` 发显式 H2D staged DMA，并加 stream 同步，会打断 Graph。所以 C 的 staged DMA 通路（仓库已有范式，§2.2.1）无论哪条路径都值得早点摸熟——路径②它就是主力搬运手段。**密切留意 A 的 P0b 结论。**

---

## 协作约定

1. **交汇就一个数据结构**：A 的输出 = B 的输入 = `top_k_device_locs`（物理槽号清单）。**第一周三人先把它的 shape 和语义敲成契约**，之后各写各的，P3 自然能接上。
2. **起跑姿势**（依赖关系决定）：
   - **A 第一周全力冲 P0a + P0b**（P0a 打通 `sgl_kernel_npu` 构建链路是写任何 kernel 的前提，P0b 门控决定架构走①还是②，结论影响全局）；
   - **B、C 不必等 A，立刻并行开干**（B 探 SFA 寻址、与 A 共享 `sgl_kernel_npu` 通路；C 搭 pool + 生命周期 + Python 参考实现 + 摸熟 staged DMA 通路）。
3. **合体节点**：
   - **P4**：A 的 kernel + B 的 SFA + C 的同步端到端 CANN Graph capture/replay，三流第一次合体一起调；
   - **P5**：多请求、异常回收、CP 路径（§2.3.2 标注的未覆盖点）按子任务再分。

## 阶段—负责人对照（P0–P5）

| 阶段 | 主负责 | 协作 |
|------|--------|------|
| P0a `sgl_kernel_npu` 构建链路 | **A** | B 共享通路 |
| P0b aclrtHostRegister 探针（门控分叉） | **A** | C 留意结论（决定路径①/②） |
| P1 SFA 寻址探针 | **B** | — |
| P2 AscendC residency kernel（路径②为 judge-only + C 的 staged DMA） | **A** | C 提供 Python 参考实现；路径②下 C 接搬运 |
| P3 SFA 路线 C 原型 | **B** | A 提供 `top_k_device_locs` |
| P4 CANN Graph 验证 | **A + B + C** | 三流合体 |
| P5 生产化（含 CP 路径） | **A + B + C** | 按子任务分 |

## 第二部分：NPU HiSparse 适配方案

> **读前须知（接 01 文档的比方）**
>
> 01 文档把 GPU HiSparse 讲成"**小书桌 + 大仓库**"：GPU 只摊开几页(hot buffer)，整本书放仓库(host 内存)，按需翻页(swap-in)。本文要做的，就是**把这套搬到华为昇腾 NPU 上**。
>
> 一句话目标：**让 NPU 也只在显存里放一张小书桌，全量 KV 挪到 host，按需搬运。**
>
> 但 GPU 那套不能照抄，因为 NPU 硬件长得不一样。全篇其实就在回答三个"搬过去会卡在哪"：
>
> 1. **谁来搬页**(§2.3.1)：GPU 用一个 warp 既判断又搬运；NPU 得把"判断"和"搬运"拆给不同的硬件单元分工。
> 2. **页码怎么对**(§2.3.2)：GPU attention 直接吃"绝对物理页码"；NPU SFA 默认 PA_BSND 吃"逻辑页码"、算子内查 block_table 翻译，但 SFA 内核源码 + CI 同样证实 BSND 一层寻址(`block_table=None`、`sparse_indices` 经 `+s2StartIdx` 后当 S 轴下标)，和 GPU 一样基本不用翻译。所以这层**不一定要改算子**——走 BSND(路线 A)即可对位 GPU。
> 3. **搬得动吗、能不能用图模式**(§2.5 / §2.8)：NPU 从 host 读数据的延迟、能否被 CANN Graph 固化，都得先探针实测。
>
> 后面的硬件细节、伪代码、CANN 引用，都是在精确地回答这三个问题。

### 2.1 现状与目标

**一句话**：NPU 现在已经会"只算 top-k"(省计算)，但 KV 还是全量堆在显存里(不省显存)。本文补上缺的那半——让 KV 也能下沉到 host。

**当前 NPU DSA 路径**（`forward_dsa_prepare_npu` + `forward_dsa_core_npu`）已有完整的 sparse attention 推理：`npu_lightning_indexer` 选 top-k → `npu_sparse_flash_attention` 做 token 级 sparse attention。但 KV cache 全量常驻 NPU（PA_BSND layout, block_size=128），只省计算量不省显存。

| 维度 | CUDA DSA | NPU DSA |
|------|----------|---------|
| Indexer | FP8 paged MQA logits + fast_topk | `npu_lightning_indexer` |
| Attention | FlashMLA sparse / tilelang / FA3 | `npu_sparse_flash_attention` |
| KV layout | flat token pool, page_size=1 | PA_BSND, block_size=128 |
| HiSparse | 有（swap-in kernel） | **无** |
| KV 常驻 | 启用 HiSparse 时不常驻 | 全量常驻 |

**核心缺失**：没有 residency runtime（hit/miss/LRU/swap-in）。"Device 访问 Host 内存"本身**昇腾官方支持**（`ACL_HOST_REGISTER_MAPPED`，A3/A2 √，原文见 §2.2.2）；缺的是 HiSparse 需要的那个更苛刻的具体用法——**AscendC kernel 内 DataCopy 直读 host**。现有 NPU 完全没走这条路：host pool 走普通 `pin_memory`、host↔device KV 搬运走显式 staged DMA（`transfer_kv_dim_exchange`），全树无任何"kernel 内直读 host"先例（§2.2.1）。所以 P0 不是验证一个未知能力是否存在，而是验证已知能力的这个用法能否成立、带宽够不够（§2.2）。

**目标**：KV cache 不全量常驻 NPU，只保留 hot buffer，全量 KV 放 host，按需 swap-in。

### 2.2 主线方案

架构 1:1 对标 GPU HiSparse：通过 `aclrtHostRegister` 让 AscendC kernel 直接读取 Host KV，在 kernel 内原子完成 hit/miss/LRU/copy，输出 hot slot 给 SFA。**寻址走路线 A（BSND、不改算子）**——hot buffer 以 BSND 布局当 `key/value` 传入、`block_table=None`、`sparse_indices` 填 hot buffer 内的物理槽号，对位 GPU 的"扁平池 + 物理槽清单"。SFA 内核源码 + CI 实测证实 BSND + token 粒度一级直址可行，无需改算子源码（依据见 §2.3.2；改算子的 `PHYSICAL_HOT_SLOT` 路线 C 仅作兜底）。

```
每步 decode (per layer):
  npu_lightning_indexer → top_k_tokens [B, 2048]
  → AscendC residency kernel:
      输入: top_k_tokens, host_kv_devPtr, hot_buffer, lru_state
      kernel 内部: hit/miss 判定 → LRU 更新 → miss DataCopy Host→hot_buffer
      输出: top_k_device_locs [B, K]（hot buffer 内物理槽号）
  → npu_sparse_flash_attention(           # 路线 A：不改算子
      key=hot_buffer, value=hot_buffer,    # BSND 布局，整块 hot buffer
      sparse_indices=top_k_device_locs,    # 直接索引 hot buffer 的 S 轴
      block_table=None, layout_kv="BSND", sparse_block_size=1)
```

**待验证前提（P0 门控）**：HiSparse 主线依赖"kernel 内直读 host"。这个能力的官方依据、两条注册路（`HostGetDevicePointer` 受限 vs `MallocHostWithCfg`+VA_FLAG 统一地址）、以及仍需上机实测的边界，集中在 **§2.2.2**。结论先行：能力官方背书，P0b 只实测"kernel 内 DataCopy 可用性 + EP 形态带宽" + "含 host read 的 kernel 能否被 Graph capture"，按结果分两条路——

| 分叉 | 条件 | 架构后果 |
|------|------|----------|
| **路径 ①：零拷贝（1:1 对标 GPU）** | kernel 内 DataCopy 直读成立、带宽可接受 | residency kernel 内"判定 + 搬运"一气呵成，上面的伪代码成立 |
| **路径 ②：staged DMA（架构改造）** | 直读不成立 或 带宽过高 | kernel 内**只算 miss list**，搬运退到 kernel 外，用既有 `aclrtMemcpyAsync` / `transfer_kv_dim_exchange` 做显式 H2D，见下方"备选架构" |

**路径②并非"失败兜底"**：判定平面本就不依赖 host 直读、可立即开写，故 P0b 从"成败门控"降级为"能否把搬运也并进 kernel 的优化"（论证见 §2.7.3.1 / §2.7.5）。

**备选架构（路径 ②，探针 1 失败时）**：

```
每步 decode (per layer):
  npu_lightning_indexer → top_k_tokens [B, 2048]
  → AscendC judge kernel（只判定，不搬运）:
      输入: top_k_tokens, device_buffer_tokens, lru_state
      输出: miss_list [B, M], evict_slots [B, M], top_k_device_locs [B, K]（hit 部分已填）
  → host 侧发起 staged H2D（既有范式）:
      aclrtMemcpyAsync(hot_buffer[evict_slots] ← host_pool[miss_list], H2D, copy_stream)
      或 sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange(..., H2D)
      回填 top_k_device_locs 的 miss 部分
  → npu_sparse_flash_attention(..., sparse_indices=top_k_device_locs, ...)
```

路径 ② 的代价：**每步 decode 多一趟 device→host→device 往返**（kernel 出来拿 miss list、发 DMA、等完成、再进 SFA），打断 CANN Graph 单图捕获。但正确性可保证、复用仓库已验证的 staged DMA 通路。两条路差别只在 Graph 友好度与一趟往返延迟，**都有 proven 退路**（详见 §2.7.5）。

#### 2.2.1 代码依据（双仓库实证）

以下结论来自两处源码审计，是上面路径①/②分叉判断的硬证据：
- **调用侧**：本仓库 `sglang-v0.5.12` 既有 NPU 后端（谁调用、怎么路由）；
- **实现侧**：外部算子包 **`sgl-kernel-npu`**（已按 sglang CI 钉定的 tag `2026.05.01` 拉取到 `../sgl-kernel-npu/`），即所有 NPU 自定义算子的真实 AscendC/host 源码。这一侧此前只能靠 grep 推断，现在是逐行读过的实证。

**(1) NPU 全程没有 host-register，且代码主动绕开它。** 分配器派发表把 NPU 显式路由到普通 pinned 内存，而非 CUDA 的 `cudaHostRegister`（[memory_pool_host.py:147](../sglang-v0.5.12/python/sglang/srt/mem_cache/memory_pool_host.py#L147)）：

```python
ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,   # CUDA 默认走 cudaHostRegister
    {
        "npu": alloc_with_pin_memory,    # NPU 走普通 pin_memory，不注册
        "musa": alloc_with_pin_memory,
    },
)
```

且 `is_pin_memory_available()` 在没有 CUDA 时直接返回 `False`（`utils/common.py:656`）。调用侧 `grep aclrtHostRegister`、实现侧 `sgl-kernel-npu` 全仓 `grep -i "HostRegister|aclrtMallocHost|ACL_HOST|svm|unified memory"` 均零功能性命中（唯一的 "zero-copy" 字样在 DeepEP MoE 的 RDMA/IPC ring-buffer，是 device↔device，与 host KV 无关）。**含义**：GPU HiSparse 的"kernel 内跨 PCIe 直读 host"模式，在 NPU 两侧都无代码先例——但"先例缺失"不等于"能力缺失"，该能力官方是支持的（§2.2.2），P0 验证的是它在 kernel DataCopy 下的具体可用性。

**(2) 既有 NPU host↔device KV 搬运 = 纯 host 侧 staged DMA，且整支算子里没有 kernel。** host 侧 KV 的搬运委托给外部算子 `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`，带 `H2D` / `D2H` 方向参数（`memory_pool_host.py:499/507/606/614`）。读它在 `sgl-kernel-npu` 里的实现（[transfer_kv_dim_exchange.cpp](../sgl-kernel-npu/csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp)）坐实了它的形态——**它根本不是 AscendC kernel，是一个纯 `HOST_API`**：

```cpp
// transfer_kv_dim_exchange.cpp（节选，H2D 分支）
auto device_indices_cpu = device_indices.cpu();   // ① 索引先拉回 CPU
auto host_indices_cpu   = host_indices.cpu();
const int64_t num_pages = device_indices.size(0) / page_size;
for (const auto i : c10::irange(num_pages)) {       // ② 在 CPU 上逐页 for 循环
    auto device_page_index = device_indices_cpu[i*page_size].item<int64_t>() / page_size;
    auto host_page_index   = host_indices_cpu[i*page_size].item<int64_t>() / page_size;
    void *device_k_ptr = device_k[0][device_page_index].data_ptr();
    void *host_k_ptr   = host_k[host_page_index][0].data_ptr();
    aclrtMemcpy2dAsync(device_k_ptr, k_device_pitch, host_k_ptr, k_host_pitch,  // ③ 每页一发 2D DMA
                       k_width, height, ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
    // device_v 同理再发一发
}
```

三个特征与 GPU HiSparse 的"kernel 内零拷贝"完全相反：① 索引必须 `.cpu()` 物化回主机；② 配对/页号计算是 **CPU 串行 for 循环**，不是 device 上并行；③ 每页一次 `aclrtMemcpy2dAsync`，用 `pitch`（device 端 `device_pages_num*page_size*…`、host 端 `page_size*…`）一次搬该页的**全部 layer**（height=`total_num_layers`），所谓 "dim_exchange" 就是 device 布局 `[layer, dev_page, page, head, dim]` 与 host 布局 `[host_page, layer, page, head, dim]` 的 layer/page 维互换。**这正是路径 ② 备选架构的现成通路**——但也说明它无法塞进单个 device kernel：搬运由 host 驱动、按页发射多次异步 DMA。

**(3) 可改的 AscendC 源码不在调用仓库,在外部包 `sgl-kernel-npu`,贡献链路已具象。** 本仓库 `sglang-v0.5.12` 全树 `grep DataCopy / __aicore__ / LocalTensor / TPipe` 零命中,所有 NPU 自定义算子来自外部 pip 包 `sgl_kernel_npu`。该包源码已拉取到 `../sgl-kernel-npu/`(tag `2026.05.01`,即 sglang `scripts/ci/npu/npu_ci_install_dependency.sh:71` 钉定的版本),结构如下：

```
sgl-kernel-npu/
├── csrc/                          # 每个算子一目录，op_host(tiling+下发) + op_kernel(AscendC kernel)
│   ├── cache_location_assign/     # ← 最接近 HiSparse 驻留表写入的现成 AscendC 算子(见下 (5))
│   ├── assign_cache_op/           # device↔device 范围拷贝 kernel
│   ├── transfer_kv_dim_exchange/  # host↔device staged DMA(纯 host_api，见 (2))
│   └── lightning_indexer/         # top_k 索引(本仓库自带 AscendC 实现，见 (4))
├── python/sgl_kernel_npu/         # python 封装 + torch.ops 注册
├── build.sh / CMakeLists.txt / cmake/   # ← P0a 要打通的本地构建入口
└── tests/                         # 算子单测，可作 P0a 验证与新算子的模板
```

**含义**：写 residency kernel = 在 `csrc/` 下新增一个算子目录(`op_host` tiling + `op_kernel` AscendC),走 `build.sh` 编译、`python/` 注册 `torch.ops.npu.*`、`tests/` 加单测,向该包提 PR;改 SFA 寻址(路线 C)= 改 SFA 算子源码(在 `ops-transformer` 仓库可读,但不在本调用仓库、也不在 `sgl-kernel-npu`,改后还要自带构建分发并确认与生产 `torch_npu` 二进制对齐,故最深)。**P0a 的"hello-world AscendC 算子"已有现成模板**:`csrc/helloworld/`。residency kernel 的源码不在调用仓库手边,需先在 `../sgl-kernel-npu/` 跑通本地构建/加载(见 §2.6 P0a 与 ASSIGNMENTS.md 中 A 的工作量)。

**(4) SFA 稀疏分页寻址是确证在用的（绿灯）。** `npu_sparse_flash_attention` 带 `layout_kv="PA_BSND"` + `block_table` + `sparse_block_size=1` + `sparse_mode=3` + `attention_mode=2` 真实在跑（[ascend_backend.py:949](../sglang-v0.5.12/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L949)），block_table 由 `req_to_token[...][:, ::page_size] // page_size` 构造（`:359`）。这给路线 A′/C 一个确定的注入点，详见 §2.3.2。配套的 `lightning_indexer`（top_k 引擎）在 `sgl-kernel-npu` 里是**自带完整实现的算子，不是壳**：`csrc/lightning_indexer/op_kernel/` 下有约 2262 行真实 AscendC kernel（`lightning_indexer_kernel.cpp` 等），经 `build.sh` 编译注册为 `torch.ops.npu.lightning_indexer`，源码就在手边、可读可改。README（[csrc/lightning_indexer/README.md](../sgl-kernel-npu/csrc/lightning_indexer/README.md)）说明它**仅 Atlas A3 支持、默认 `sparse_count=2048`、key 支持 `PA_BSND` PageAttention 布局**——印证 top_k→稀疏寻址这条链在 A3 上是成熟能力，不需我们重写。（关于 SFA 自身源码：它不在 `sgl-kernel-npu`，但**真源码在另一仓库 `ops-transformer/attention/sparse_flash_attention`、已可读**——见 §2.3.2。所以"改 SFA 够不着"的旧说法需修正为"源码可读、但改它需自带算子构建分发，仍是最被动路线 C"。）

**(5) 现成 AscendC 算子 `cache_location_assign` 给出了 residency kernel 的真实编码范式。** 这是整个包里语义最接近 HiSparse 驻留逻辑的 AscendC kernel（[cache_loc_assign_kernel.cpp](../sgl-kernel-npu/csrc/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp)）——它做 token-pool↔cache-location 的索引映射(纯 bookkeeping，无 KV 数据搬运)。读它的实现把 §2.7 伪代码里几个抽象点全部落到了可对照的真实写法,**也暴露了 NPU 的硬约束**：

| §2.7 设计点 | `cache_loc_assign` 的真实写法 | 对 residency kernel 的含义 |
|--|--|--|
| 并行单元 | `coreId = AscendC::GetBlockIdx()`,按 row 切给各 vector core(`rowNumNoTail`+`tailNum` tiling) | 是 **per-core 分块**,不是 GPU 的 per-request warp;一个 core 串行处理分到的多行 |
| UB 管理 | 显式 `TPipe` + 7 个 `TBuf<VECCALC>` + `TQue<VECIN,BUFFER_NUM=2>` 手工分配 | 印证 §2.7.4 的 UB 预算必须手算,double-buffer 要显式声明 |
| H2D/UB 搬运 | `DataCopy(ub, gm, count)` + `DataCopyPad(gm, ub, DataCopyExtParams{...})` | MTE 搬的是**规则块/对齐 pad**,不是任意索引 scatter |
| pipe 同步 | 手动 `SetFlag/WaitFlag<HardEvent::MTE2_V>`(读完再算)、`<HardEvent::V_MTE3>`(算完再写) | 印证 §2.8 的 event 模型:流水级间靠 HardEvent 显式卡 |
| **按索引 scatter/gather** | **`for j: cache=ubCacheLoc.GetValue(idx+j); tokenPoolLocal.SetValue(j, cache)`** —— 逐元素标量 `GetValue/SetValue` | **决定性约束**:NPU 没有 warp 并行 scatter,"按 miss_list 把零散 token 写到 hot slot"在 kernel 内退化为 **Scalar 单元逐元素串行循环**(§2.7.5 的瓶颈论断由此坐实);这正是 hit 判定必须交给 Vector 批量比较、而非 Scalar 逐个比的根因 |

**含义**:residency kernel 不是从零摸索——`cache_location_assign` 已经示范了 tiling/UB/DataCopy/HardEvent 的全套骨架,新算子可直接以它为模板。但它也用最直白的方式证明了 §2.3.1"难点①谁来搬页"的本质:**逐元素 `SetValue` 串行**就是 NPU 没有 SIMT scatter 的代价,因此判定(Vector 批量)与搬运(MTE 整块)必须拆开、再各自规避标量循环。

#### 2.2.2 官方文档依据（CANN 8.5 acl API）

**查证 CANN 8.5 官方 acl API 文档：'Device 读写 registered host 内存'是官方明文支持的能力，不是未知数。** 以下为逐条原文依据（文档可经本地 `curl` 访问；WebFetch 因 claude.ai 域名校验被企业策略挡，本地网络路径不挡）：

| API / 类型 | 官方原文（节选） | 对 P0b 的意义 |
|--|--|--|
| `aclrtHostRegisterType`（枚举） | `ACL_HOST_REGISTER_MAPPED = 0` // "the host memory is mapped and registered as **accessible to the device, including read and write**" | "device 可读写 host 内存"被官方明文背书——这是路径①的能力基础 |
| 产品支持矩阵 | Atlas A3 训练/推理 √、A2 训练/推理 √（200I/500 A2、纯推理卡 ☓） | 正好覆盖目标硬件 A3/A2 |
| `aclrtHostRegister` / `aclrtHostRegisterV2` | "Maps and registers the host memory as a memory address that can be accessed by the device" | host 侧注册入口；V2 配 `ACL_HOST_REG_MAPPED`(0x2) / `ACL_HOST_REG_PINNED` |
| `aclrtHostGetDevicePointer` | "**Obtains the device memory address** registered and mapped by `aclrtHostRegisterV2`"，签名 `(void *pHost, void **pDevice, uint32_t flag)`：输入 host 指针、输出 device 指针 | **命名易误读**:它不是"在 device 取 host 指针"，而是**host 侧的地址翻译器**——把 host 虚拟地址翻成 AI Core 能用的 device 地址,再作 kernel 入参(对位 CUDA `cudaHostGetDevicePointer`) |

**正确的使用流程**（host 侧准备 → device 侧 kernel 消费）：

```
# host 侧（CPU，准备阶段）
ptr = aclrtMallocHost(...) 或 malloc(...)            # 一块 host 内存（4K 页对齐）
aclrtHostRegisterV2(ptr, size, ACL_HOST_REG_MAPPED)  # 注册为 device 可访问
aclrtHostGetDevicePointer(ptr, &devPtr, 0)           # 翻译出 device 视角地址 devPtr
launch kernel(devPtr, ...)                           # devPtr 作为入参喂给算子
# device 侧（AI Core，kernel 内）
GlobalTensor.SetGlobalBuffer(devPtr); DataCopy(UB, gm, ...)  # ← 读的其实是那块 host 内存
```

**仍需 P0b 实测的，是一条带官方限定的边界**——两条注册路对"能否做 memory operations"给了**相反**的措辞：

| 路 | 拿到的 devPtr | 官方原话 |
|--|--|--|
| **路 A**：`HostRegisterV2` + `HostGetDevicePointer` | 专门的映射地址 | "This address **cannot** be used for memory operations, such as memory copy." |
| **路 B**：`aclrtMallocHostWithCfg` + `ACL_RT_MEM_ATTR_VA_FLAG` / `vaFlag=1` | **与 host 地址相同**的统一地址 | "the mapped device address is the same as the host address and **can** be used for memory operations." |

"memory operations / memory copy"是否涵盖 **AscendC kernel 内的 `DataCopy`**，官方未明说——这是 P0b 唯一要上机验证的点，**初步判断路 B（统一地址）比路 A 更可能是路径①正解**。另注 `aclrtMallocHostWithCfg` 提到 **RC 形态**（host/device 集成）下"分配 host 内存等同于分配 device 内存"，但目标 A2/A3 是 **EP 形态**（经 PCIe/HCCS 分离），统一地址在 EP 上的 kernel 内可用性与跨总线带宽是文档给不出、必须实测的部分。P0b 的定性影响见 §2.7.5。

> **官方文档 URL**（CANN Commercial 8.5.0，acl API C）：
> - [aclrtHostRegisterType 枚举](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1840.html)
> - [aclrtHostRegister](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1804.html) · [aclrtHostRegisterV2](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_2128.html)
> - [aclrtHostGetDevicePointer](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_2129.html)
> - [aclrtMallocHostWithCfg](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1799.html)

### 2.3 差异与对策

#### 2.3.1 swap-in 机制 → AscendC kernel 直读 Host

> **难点①：谁来搬页。** GPU 上一个 warp 的 32 个 lane 像 32 只手，一边判断"这页在不在桌上"、一边各自把缺的页抓上来，手脚不分家。NPU 没有这种"既能判断又能搬运的一队手"——它把活拆给不同工种：**Vector 单元负责快速比对(判断)、Scalar 单元负责调度派活、MTE 引擎(DMA)负责搬运**。所以同一件事，NPU 要改写成"先一批判断完，再流水线搬运"的两阶段分工。

| | GPU | NPU |
|--|--|--|
| Host 内存注册 | `cudaHostRegister` + UVA | `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer`（A3/A2 官方支持 device 读写，§2.2.2），或 `aclrtMallocHostWithCfg`+VA_FLAG 统一地址 |
| kernel 内读 Host | PTX `ld.global.nc`（texture cache streaming） | DataCopy via devPtr（能力官方背书；**P0b 仅实测 kernel 内 DataCopy 的具体可用性+带宽**：通过→零拷贝；不通过→退 staged DMA，§2.2.2） |
| 既有 host↔device 搬运范式 | kernel 内零拷贝直读 | 显式 staged DMA（`transfer_kv_dim_exchange` H2D/D2H，§2.2.1） |
| kernel 源码位置 | 本仓库 `hisparse.cuh` 可读可改 | 外部包 `sgl-kernel-npu`(已拉取 tag 2026.05.01),以 `csrc/cache_location_assign/` 为编码模板,需先打通构建(§2.2.1 / P0a) |
| 并行模型 | SIMT：32 lanes 各读不同地址，隐式同步 | SIMD：MTE DMA 搬整块，Scalar core 做判定 |
| Graph 兼容 | CUDA Graph 天然支持 | CANN Graph 待验证（路径②的 H2D 往返会打断单图捕获） |

**SIMD vs SIMT 实现差异**：GPU 用一个 warp 同时做"判定 + 搬运"（32 lanes scatter/gather）；NPU 把两件事分到不同 pipe——**Scalar** 收集 miss list / LRU / 地址计算 / 派发，**MTE** 搬运 Host→UB→hot_buffer（DMA 搬连续块，适合整 token），**Vector** 批量比较做 hit 判定（必须，非可选——Scalar 弱）。算法逻辑照搬 GPU（hit 判定、LRU、miss 配对），瓶颈从"scatter"转移到"hit 判定的并行度"，miss copy 不会比 GPU 差。完整算法见 §2.7。

#### 2.3.2 attention 寻址 → 让 SFA 读 hot buffer

> **难点②：页码怎么对。** attention 按清单取 KV，问题是清单号码指向哪个池子。快递柜比方：GPU FlashMLA 清单写**绝对格子号**(物理扁平行号，调用方算好)，kernel 拿到 `idx/page_size`、`idx%page_size` 拆回分页坐标即取、**从不查登记簿**；NPU SFA 有两种收件模式——`PA_BSND` 清单写**订单号**、算子内查 block_table 翻成格子号(多一层)，`BSND`(`block_table=None`) 清单当**格子号**索引 key 池 S 轴(一层)。所以把 residency kernel 产出的格子号交给 SFA 有三条路：走 BSND 直吃格子号(路线 A，不改算子)、保持 PA_BSND 但重映射对照表到 hot buffer(路线 A′)、改算子内部(路线 C，兜底)。**SFA 内核源码 + CI 实测证实 BSND 一层寻址是被测的一等路径，GPU 的"零修改"红利 NPU 拿得到。** 一个 NPU 侧的限定：SFA 的清单号是**相对窗口基址**的（kernel 内 `s2Idx = sparse_indices + s2StartIdx`），不像 FlashMLA 是绝对行号，接入时要处理这个基址（见下方路线 A 注）。

| | GPU FlashMLA | NPU SFA |
|--|--|--|
| 寻址模式 | `indices` = **绝对物理扁平行号**（唯一模式）；kernel 内 `idx/page_size`、`idx%page_size` 拆分页坐标，**不解引用 block_table** | **两种布局可选**：`PA_BSND`（`sparse_indices` 逻辑 + `block_table` 翻译，两层）或 `BSND`（`block_table=None`，`sparse_indices` 经 `+s2StartIdx` 后作 key 池 S 轴下标，一层、对位 GPU） |
| logical→physical 翻译 | kernel **外部**（Python 侧 `block_table[abs//bs]*bs+...` 算好）；sparse 路径的 `block_table` 是传了也忽略的 vestigial 参数 | PA_BSND 在 kernel 内查表；**BSND 无 block_table 翻译层**（kernel 内 `block_table` 不绑定 GM） |
| 索引基准 | 绝对（无基址相加） | **窗口相对**：`s2Idx = sparse_indices + s2StartIdx`，sparse 场景 `s2StartIdx` 可能非 0 |
| HiSparse 需要改 attention 吗 | 不需要 | **不一定需要**——见下方三条路线，BSND 直填 hot slot 可不改算子 |
| 稀疏粒度 | token（page_size=1） | **arch35（950 类）硬编 token-wise `sparse_block_size=1`**；block-wise(2~128) 仅 arch22 |

**依据(从"官方文档"升级为"算子内核源码 + CI 实测"——SFA 真源码在 [`ops-transformer/attention/sparse_flash_attention`](../ops-transformer/attention/sparse_flash_attention))**：
- 对应算子是 `SparseFlashAttention`(非 `SparseFlashMla`)：[`op_host/sparse_flash_attention_def.cpp`](../ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp) 的输入/attr 列表与调用点 `torch_npu.npu_sparse_flash_attention(...)` 逐字段吻合(query/key/value/sparse_indices/block_table/query_rope/key_rope + 同名 attr)。
- **两种寻址模式由编译期 `isPa`(=`layout_kv==PA_BSND`) 选**，第一手读自 [`sparse_flash_attention_service_vector_mla.h:225-237`](../ops-transformer/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h#L225)：PA_BSND 走 `blockTableGm.GetValue(...)*blockSize + s2Idx%blockSize`(两级)；BSND 走 `realkeyOffset = boIdx*s2Size + s2Idx`(一级直址，`block_table` 连 GM 都不绑定)。
- **索引是窗口相对的**：`s2Idx = sparseIndicesGm.GetValue(...) + runInfo.s2StartIdx`(同文件 `:207`)；`util_regbase.h:70` 注释明写 `s2StartIdx /* s2的起始位置，sparse场景下可能不是0 */`。这是与 FlashMLA(绝对行号)的真实差异。
- **BSND(`block_table=None`) 与 PA_BSND 都是 CI 实测的一等路径**：[`sparse_flash_attention_paramset.py`](../ops-transformer/attention/sparse_flash_attention/tests/pytest/sparse_flash_attention_paramset.py) 的 `ENABLED_PARAMS` 含 `bsnd_basic`/`bsnd_multi_batch`/`pa_bsnd`/`tnd_*`，**全部 `sparse_block_size=1`**(token-wise)。
- **粒度约束**：arch35（950 类）`sparse_flash_attention_kernel_mla.h:427` 硬编 `constInfo.sparseBlockSize = 1`，只支持 token-wise；block-wise(2~128) 仅 arch22 实现。
- 散读由 SFA 自身承担并优化(算子文档称内部已做"指令缩减及搬运聚合")，**不是 HiSparse 需新解决的瓶颈**。
- 约束栏：`return_softmax_lse=True` 时不支持图模式；HiSparse 用 False，不受限。BSND 要求 `layout_query==layout_kv`（[`check_valid_param.py:51`](../ops-transformer/attention/sparse_flash_attention/tests/pytest/check_valid_param.py#L51)）。

**为什么 SFA 默认用 PA_BSND 内部翻译**（设计选择，非缺陷）：减少 kernel launch、DMA 紧耦合、按 block 连续搬对 SIMD 友好。但 SFA 内核同时实现了 BSND 一层寻址（`isPa=false` 分支），给 HiSparse 留了"不改算子"的口子。

**三条寻址路线（按改动深度排序；此前文档基于"NPU 散读是死结、必须自己改寻址"把路线 C 列为推荐，内核源码 + CI 推翻了该前提——token-wise 是 arch35 唯一支持、散读已被算子优化、BSND 一层寻址是实测路径，故推荐降级为路线 A）**：

- **路线 A（推荐·不改算子）**：hot buffer 以 **BSND** 布局当 `key/value` 传入（`block_table=None`），`sparse_indices` 填 hot buffer 内槽号。BSND 一级直址是 SFA 内核 + CI 实测路径,**无需碰算子源码**。这是对位 GPU"扁平池 + 物理槽清单"的最干净路径。**两个约束**：① BSND 要求 `layout_query==layout_kv`（`check_valid_param.py:51`），现网是 `layout_query="TND"`+`layout_kv="PA_BSND"`，**故须把 query 一并切到 BSND**（query 侧 reshape/打包也要改，非纯 kv 改动）；② kernel 内 `s2Idx = sparse_indices + s2StartIdx`，**槽号不是绝对填入**——须确认 decode 场景 `s2StartIdx=0`，或在喂入前扣除该基址。两点都纳入 P1 验证。
- **路线 A′（不改算子·页粒度）**：保持 **PA_BSND**,`key/value` 传 hot buffer 池、**重映射 `block_table`** 指向 hot buffer block,`sparse_indices` 仍填逻辑位。SFA 内部照常翻译、落到 hot buffer。代价：页粒度（block_size 的整数倍驻留）。`sparse_block_size` 仍为 1,**不受 arch35 token-wise 限制影响**。
- **路线 C（改算子·最后手段）**：给 SFA 新增 `PHYSICAL_HOT_SLOT` 模式跳过 block_table。**仅当路线 A/A′ 经 P1 验证不可行时才动它**。注意:SFA 源码现已可读(就在 `ops-transformer`),改算子门槛从"看不见源码的盲改"降为"源码可读、但需先确认生产 `torch_npu` 二进制与此仓版本对齐 + 自带算子编译分发链路"——降了一档,但"可读"不等于"已到手",仍是最被动选项。

**SFA 算子实际调用点**（`ascend_backend.py`，`forward_sparse` 路径）：

| 调用点 | 方法 | 触发条件 | 说明 |
|--------|------|---------|------|
| `ascend_backend.py:949` | `forward_sparse`（主路径） | 默认（非 CP / decode） | 单次 SFA，标准稀疏 attention |
| `ascend_backend.py:739` | `do_cp_balance_attn`（prev 半） | prefill + `is_nsa_enable_prefill_cp()` + `attn_cp_size > 1` | Context Parallel zigzag 切分 |
| `ascend_backend.py:761` | `do_cp_balance_attn`（next 半） | 同上 | 另一半 |

三处共用同一个 `torch_npu.npu_sparse_flash_attention` 算子，当前参数一致（`sparse_block_size=1`, `layout_kv="PA_BSND"`, `sparse_mode=3`, `attention_mode=2`, 都带 `block_table`）。路线 A/A′ 只改 Python 侧的调用参数——路线 A 把 `layout_kv` 切到 `"BSND"`、`block_table=None`、喂小 hot buffer 当 `key/value`、`sparse_indices` 填 hot slot；路线 A′ 保持 `PA_BSND` 但重映射 `block_table`。两者**三处调用都自然受益、且不碰算子源码**；仅路线 C 需改算子内部寻址分支。

**CP 路径需单独验证**：`do_cp_balance_attn` 把序列切两半，每半有独立的 `topk_indices_prev/next` 和 `actual_seq_lengths_kv`。HiSparse 的 hot buffer / residency 状态按整序列管理，切半后两个 SFA 调用如何共享 hot buffer、residency kernel 在 CP 下跑几次，是当前方案未覆盖的点。**第一阶段（P0–P3）聚焦非 CP 主路径（`:949`），CP 路径留到 P5**（见 §2.6）。

#### 2.3.3 内存池 → 新增 HiSparse 专用组件

不整改现有 `NPUMLATokenToKVPool`，新增一套：

| GPU 组件 | NPU 对应（新增） | 职责 |
|----------|-----------------|------|
| `HiSparseNSATokenToKVPool` | `NPUHiSparseMLATokenToKVPool` | Device hot buffer，**布局由寻址路线定**：路线 A 用 BSND（`sparse_indices` 直填 hot slot，§2.3.2），路线 A′ 用 PA_BSND + 重映射 block_table；只分配 `device_buffer_size` slots |
| `HiSparseTokenToKVPoolAllocator` | `NPUHiSparseAllocator` | logical↔device 映射，staging/decode 生命周期 |
| `MLATokenToKVPoolHost` + `cudaHostRegister` | `NPUMLATokenToKVPoolHost` + `aclrtHostRegisterV2`（§2.2.2） | Host pool，全量 KV，注册后 kernel 可直读 |

不动的：现有 pool/allocator、PA_BSND layout、`index_k_buffer`（常驻 NPU，不 offload）。

#### 2.3.4 Indexer KV 常驻

Indexer KV 维度小，全量常驻 NPU：

```
indexer_kv: 128K × 256 bytes × 61 layers = 1.95 GB（占 attention KV 的 ~22%）
attention_kv: 128K × 1152 bytes × 61 layers = 8.6 GB（需要 offload）
```

`npu_lightning_indexer` 调用不变。

### 2.4 请求生命周期（NPU 完整流程）

> **导读**：本节把 01 §2.2 的四阶段生命周期(prefill → staging → decode → finish，"写满大桌 → 整本归档换小桌 → 边写边翻页 → 收工清桌")逐字翻成 NPU API——概念不变，只是 CUDA 调用换成 AscendCL(`cudaMemcpyAsync`→`aclrtMemcpyAsync` 等，对照表见 §2.8.1)。对着 01 §2.2 看即可。

对标 GPU HiSparse 的 prefill → staging → decode → finish 四阶段，NPU 侧完整流程如下。**本节描述 PD 混合（prefill 与 decode 在同一实例）场景**；PD 分离场景的差异见 §2.4.1。

```
┌─────────────┐     ┌──────────┐     ┌──────────────────────┐     ┌────────┐
│   Prefill   │────>│ Staging  │────>│   Decode (per step)  │────>│ Finish │
└─────────────┘     └──────────┘     └──────────────────────┘     └────────┘
```

**Prefill**：

1. KV 写入 NPU 全量 pool（现有 `forward_dsa_prepare_npu` 路径不变）
2. Prefill 结束后，在 backup stream 上发起异步 D2H：`aclrtMemcpyAsync` 全量备份到 Host pool（`NPUMLATokenToKVPoolHost`）
3. `aclrtRecordEvent(prefill_done_event, backup_stream)`

**Staging**：

1. Poll `aclrtQueryEventStatus(prefill_done_event)` 等待 D2H 完成（或 `aclrtSynchronizeEvent` 阻塞等待）
2. 释放全量 pool 中该 request 的 slots（NPU 显存回收）
3. 分配 hot buffer：从 `NPUHiSparseMLATokenToKVPool` 分配 `device_buffer_size` 个 slots + 1 reserved slot
4. 初始化 per-request 状态：
   - `device_buffer_tokens[layer, req, slot]` 跟踪表（全 -1）
   - `lru_state[layer, req, :]` 初始化
   - `full_to_hisparse_device_mapping` 建立

**Decode（每步）**：

```
prepare_for_decode():
  │
  ├── seq_lens += 1
  ├── alloc_decode() → new out_cache_loc
  │
  └── map_last_loc_to_buffer():
        ├── backup_previous_token():
        │     在 backup_stream 上：
        │       aclrtStreamWaitEvent(backup_stream, main_stream_event)
        │       aclrtMemcpyAsync(host_pool[prev_pos], hot_buffer[prev_slot], D2H)
        │       aclrtRecordEvent(backup_done_event, backup_stream)
        │
        └── 映射 newest token → reserved slot (buffer_size 位置)

forward_decode() (per layer):
  │
  ├── main_stream wait backup_done_event
  │
  ├── 写 KV cache（newest token 写入 reserved slot）
  │
  ├── npu_lightning_indexer → top_k_tokens [B, 2048]
  │
  ├── AscendC residency kernel（详见 §2.3.1）:
  │     输入: top_k_tokens, host_kv_devPtr, hot_buffer, lru_state
  │     输出: top_k_device_locs [B, K]
  │
  └── npu_sparse_flash_attention(           # 路线 A：不改算子
        key=hot_buffer, value=hot_buffer,    # BSND 布局
        sparse_indices=top_k_device_locs,    # 直接索引 hot buffer 的 S 轴
        block_table=None, layout_kv="BSND", sparse_block_size=1)
```

**Finish**：

1. 释放 Host pool 中该 request 的 slots
2. 释放 hot buffer slots（归还 `NPUHiSparseAllocator`）
3. 清理 per-req 跟踪状态（device_buffer_tokens, lru_state, mapping）

**与现有路径的修改点汇总**：

| 组件 | 现有行为 | HiSparse 后 |
|------|---------|-------------|
| KV 写入 | 写全量 pool | 写 hot buffer reserved slot + 异步 D2H backup |
| Indexer | `npu_lightning_indexer` | 不变 |
| Attention 输入 | 全量 kv_cache + sparse_indices + block_table | hot_buffer + top_k_device_locs |
| 新增 | — | AscendC residency kernel + backup stream 同步 |

#### 2.4.1 PD 混合 vs PD 分离的生命周期差异

§2.4 描述的是 **PD 混合**场景。**PD 分离**（prefill 在 P 节点、decode 在 D 节点）下，D 节点的生命周期**没有 Staging 阶段**。

根因：Staging 阶段的全部工作是"把 device 上的全量 KV 备份到 Host"（D2H copy）。PD 混合下 KV 是本地 prefill 算出来的、人在 device 上，必须自己搬；PD 分离下 KV 由 P 节点经 RDMA **直接写进 D 节点的 Host pool**，D 节点的 device 上从未出现过全量 KV，所以无备份可做、无 device 余量可释放。

| 阶段 | PD 混合（同一实例） | PD 分离（D 节点） |
|------|--------------------|-------------------|
| KV 来源 | 本地 prefill，全量常驻 device | P 节点 RDMA 直接写 D 节点 Host pool |
| device 是否有过全量 KV | 有 | **无**（只分配 logical 索引 + Host 目标） |
| **Staging（D2H 备份）** | **有**——staging 的核心开销 | **跳过** |
| 释放 device 余量槽位 | 有 | **无**（未占用过） |
| hot buffer 初始状态 | prefill 已填满，标记为 resident | **空**——需修正 |
| 准入触发点 | prefill 输出处理后 | D 节点 KV 传输完成后 |

GPU 侧对应实现：PD 混合走 `admit_request_into_staging`（含 D2H 备份），PD 分离走 `admit_request_direct`（跳过备份）。

**PD 分离的关键正确性修正**：`alloc_device_buffer` 默认把 hot buffer 标记为"已 resident"（`device_buffer_tokens = [0..buf_size-1]`），这在混合场景成立（prefill 真填满了），但 PD 分离下 buffer 是空的，必须修正：

- 短序列（seq_len ≤ device_buffer_size）：从 Host pool 预加载填满 buffer（`_preload_to_device_buffer`）
- 长序列：把 `device_buffer_tokens` 全置 -1，让 residency kernel 首步全 miss、从 Host 加载

漏掉此修正，decode 首步会读到 hot buffer 中的垃圾数据。

**NPU 适配影响**：

1. PD 分离需 P 节点把 KV 写入 D 节点经 `aclrtHostRegister` 注册的 Host pool（RDMA / `aclrtMemcpy`），D 节点准入路径跳过 D2H 备份、只建 hot buffer + 修正 resident 标记。
2. 优先做 PD 混合（§2.4），PD 分离作为后续场景（GPU 侧 dsv4 的 direct-to-host 路径目前仍是 `NotImplementedError`，仅 NSA 路径支持）。

#### 2.4.2 prefill 长度 vs hot buffer 大小的两种场景

设 `device_buffer_size = 4k`（hot buffer 容量）。staging 时 `alloc_device_buffer` 按 prefill 长度走两条分支（以下为 NSA 路径，`hisparse_coordinator.py:288-333` / `hisparse_memory_pool.py:240-276`）。

**两个尺寸常量**：

- `device_buffer_size`：hot buffer 标称容量（如 4k），decode 时 swap-in 的 LRU 缓存大小
- `padded_buffer_size = device_buffer_size + page_size`：实际分配量，多出的一页是写**当前新生成 token** 的 reserved slot（长序列时新 token 写在这页，不占 LRU 缓存区）

##### 场景 A：prefill ≤ 4k（短序列）

整个序列装得进 hot buffer，**无需 swap-in**。

| 步骤 | 行为 |
|------|------|
| staging 分配 | `alloc_size = 页对齐(prefill_len)`，**只分配实际需要的量**，不浪费到满 4k（`:297-300`） |
| device 留存 | 全部 prefill token 都留在 device，全 resident |
| decode | indexer 选 top-k → swap-in kernel 走 fast path 直接返回 device locs，**不读 Host**（全命中） |
| 序列增长 | 每步 decode 若 `seq_len` 超过当前 buffer 容量，`_grow_device_buffers` 动态扩容（页对齐），直到触顶 `device_buffer_size`（`:335-406`，仅对 `seq_len ≤ device_buffer_size` 的 req 扩容） |
| 触顶 | 当 `seq_len` 达到 `device_buffer_size`，分配量提升到 `padded_buffer_size`（加上 reserved page），此后转入场景 B 的行为 |

PD 分离直连场景下短序列还需 `_preload_to_device_buffer` 从 Host 预热填满 buffer（见 §2.4.1）。

##### 场景 B：prefill > 4k（长序列）

序列装不下，**必须 swap-in**，hot buffer 当 LRU 缓存用。

| 步骤 | 行为 |
|------|------|
| staging 分配 | `alloc_size = padded_buffer_size`（4k + reserved page，`:301-302`） |
| device 留存 | 从 prefill 占用的全量槽位中**保留前 `device_buffer_size`（4k）个**作初始内容，其余 `free_hisparse_indices` 释放（`hisparse_memory_pool.py:250-251`） |
| 初始标记 | `device_buffer_tokens = [0,1,...,4095]`（`:328-330`），告诉 swap-in kernel buffer 初始装的是 token 0~4095 |
| decode | 每步 indexer 选 top-k → swap-in kernel 判定 hit/miss → miss 从 Host 搬入、按 LRU 踢出旧 token → 输出 `top_k_device_locs` |
| 新 token | 每步生成的新 token 写入 reserved slot（`device_buffer_size` 位置，`map_last_loc_to_buffer:458-460`），并异步增量备份到 Host（§2.4 decode 流程） |

**初始保留"前 4k"而非"最重要 4k"**：staging 时 decode 尚未开始、没有 query，无法判断 token 重要性，故按序列顺序取前 4k。这只是 LRU 缓存的初始值，decode 第一步 indexer 即按真实 query 选 top-k，miss 的从 Host 换入——一两步内 buffer 内容就被实际需要的 token 替换。

**冷启动代价**：第一步 decode 的 top-k 与"前 4k"初始内容大概率不重合（尤其序列尾部 token，局部性最强却不在 buffer），首步 miss 较多、有一批 Host 搬运。因相邻 decode 步 top-k 高度重叠，一两步后命中率回升至稳态。NPU 上 Host 读延迟若偏大，此首步 penalty 被放大，可考虑 staging seeding 优化（保留尾部/sink 混合而非纯前 4k），见 §2.5。

### 2.5 性能风险

风险分两类：**增量**（HiSparse 新引入的，是真赌注）和**存量**（SFA/现有路径本就承担的，HiSparse 基本继承、不更差）。

**增量风险（HiSparse 新引入，按优先级）**：

| 风险 | 影响 | 缓解 / 实测项 |
|------|------|------|
| **Host 读带宽/延迟（头号赌注）** | EP 形态下每个 miss 跨 PCIe/HCCS 搬 KV；带宽不够则 miss penalty 吃掉稀疏化收益 | **P0b 必测带宽**；判定平面不依赖它（§2.7.5）；MTE 批量/double-buffer 掩盖 |
| HBM 散搬 miss（无 SG-DMA 银弹） | 公开 Ascend C 接口无"GM 按索引表批量 gather"，散搬只能逐段 `DataCopy`（§2.7.1） | 排序后合并连续段（`lightning_indexer` 现成向量排序）；或粗化到 block 粒度 |
| 长序列冷启动 | 首步 buffer 初始"前 4k"与真实 top-k 不重合、miss 多（§2.4.2）；Host 读延迟大时放大 | staging seeding：保留尾部 recent + 头部 sink 混合；或 staging 末尾预热 |
| 小池子 BSND/重映射 block_table 是否被 SFA 接受 | 路线 A/A′ 成立的前提（§2.3.2）。**SFA 内核源码 + CI paramset 证实 BSND 一级直址是实测路径**：`realkeyOffset = boIdx*s2Size + s2Idx`（`service_vector_mla.h:233`），`bsnd_basic` 在 `ENABLED_PARAMS` 里、`sparse_block_size=1`。**剩余未验证点有三**：① 把 key 从全量序列换成 device_buffer_size 的小 hot buffer（公式里只是 `s2Size` 变小）；② BSND 要求 `layout_query==layout_kv`，须把 query 从现网 TND 一并切到 BSND；③ 索引含 `s2StartIdx` 窗口基址，槽号非绝对填入 | **P1 上机验证**（一级直址已有内核 + CI 先例；验证小池子变体 + query 转 BSND + s2StartIdx 处理，风险显著低于此前判断） |

**存量风险（SFA 本就承担，非 HiSparse 新增）**：

| 项 | 说明 |
|------|------|
| SFA 离散访存 | 现有 SFA 已在 `sparse_block_size=1` 下散读 2048 token，算子文档 + 内核源码表明内部已做"指令缩减 + 搬运聚合"优化。**HiSparse 把散读范围从全量 KV 缩到 hot buffer，不会更差**——此前把它列为 HiSparse 赌注是误判（§2.3.2） |
| hit 判定并行度 | 纯 Scalar 线性扫 4096 slot 慢；必须 Vector 向量化（排序求交，§2.7.3.1），这是新 residency kernel 自身的实现要求 |

### 2.6 分阶段落地

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0a: `sgl-kernel-npu` 贡献链路** | 在已拉取的 `../sgl-kernel-npu/` 跑通 `build.sh`,以 `csrc/helloworld/` 为模板新增并加载一个自定义 AscendC 算子(本仓库无源码,§2.2.1) | helloworld 算子编译通过 + `torch.ops.npu` 可调用 + `tests/` 单测过 |
| **P0b: host 直读探针** | kernel 内 DataCopy 能否直读 registered host + 带宽 —— 决定搬运并进 kernel（①）还是退 staged DMA（②）；**非全局门控**，判定平面不依赖它（§2.7.5） | 最小 kernel + 带宽数据 + 路径结论 |
| **P1: SFA 寻址探针** | 验证不改算子的路线 A/A′ 是否成立：BSND 小 hot buffer + `sparse_indices` 直填 hot slot（路线 A），或 PA_BSND + 重映射 block_table（路线 A′）——SFA 能否正确读到（§2.3.2） | 单层 golden test：路线 A/A′ 与 full-resident 对齐 |
| **P2: AscendC residency kernel** | hit/miss/LRU/copy 全流程（路径②则为 judge-only kernel + host 侧 staged DMA） | 对齐 Python 参考实现 |
| **P3: 寻址路线定稿** | P1 通过则用路线 A/A′（不改算子）做多层 decode 对齐；P1 不通过才退路线 C（改 SFA 内部寻址 ABI） | 多层 decode 对齐 full-resident |
| **P4: CANN Graph 验证** | 端到端 Graph capture/replay（路径② Graph 会被 H2D 往返打断，需评估） | 性能 benchmark |
| **P5: 生产化** | 多请求、生命周期、异常回收、CP 路径（`do_cp_balance_attn`） | 压力测试 |

P0a（外部包贡献链路）是 P2/P3 写算子的前提，必须先做。P0b 决定 P2 的 kernel 形态（路径①一气呵成 vs 路径② judge-only + staged DMA），但**判定平面可不等 P0b 先行开写**（§2.7.5）——P0b 通过则把搬运并入，不通过则保持 judge-only。

### 2.7 AscendC Residency Kernel 算法设计

§2.3.1 给出了 SIMD vs SIMT 的概念差异，本节展开到可据此编码的粒度。**调研依据**：基于 CANN 官方文档对 AI Core 硬件架构、指令队列、DataCopy 能力的描述（链接见 §2.7.6）。

> **前置说明**：本节伪代码是要落地为**向 `sgl-kernel-npu` 贡献的新算子**的目标逻辑（源码不在本仓库、需先完成 P0a，§2.2.1 (3)），不是仓库内可直接编辑的文件。以下 Phase 1–3 描述 **路径①（kernel 内零拷贝）** 形态；若走路径②，Phase 2 的 miss copy 移出 kernel、改 host 侧 staged DMA，kernel 退化为 judge-only（§2.2）。

> **导读：本节解决难点①的工程落地。** GPU 一气呵成的 swap-in kernel，到 NPU 要拆成"**先批量判断(Phase 1)，再流水线搬运(Phase 2)，最后输出格子号(Phase 3)**"。根因是两条硬件约束：① **Scalar 单元很弱**(官方称 "mini CPU")，批量比对"哪些页在桌上"必须交 Vector；② **DataCopy 不支持"给一串地址随机抓"**，只能搬规则连续块，故一批 miss 只能逐页发、用 double buffer 重叠掩盖延迟。这两条直接决定 §2.7.1 的单元分工与 §2.7.3 的三阶段写法。

#### 2.7.1 硬件执行模型

AI Core（Atlas A3/A2）的相关执行单元：

| 单元 | 角色 | 官方描述 |
|------|------|----------|
| Scalar Unit | 地址计算、循环控制、分支判定、向其他 queue 派发指令 | "mini CPU"，计算能力弱，官方建议 minimize scalar work |
| Vector Unit | SIMD 向量计算（批量比较、掩码） | 高吞吐并行计算 |
| MTE2 | GM → UB 搬运 | DMA 引擎 |
| MTE3 | UB → GM 搬运 | DMA 引擎 |

5 条独立指令队列（Vector / Cube / MTE1 / MTE2 / MTE3），**同 queue 内顺序执行，跨 queue 并行**，用 `SetFlag` / `WaitFlag` 做跨 queue 同步。Atlas A3/A2 是 Separation mode：Vector core 和 Cube core 分离，各有独立 Scalar Unit。

**关键能力边界**：`DataCopy` 只支持规则 stride（`DataCopyParams{blockCount, blockLen, srcStride, dstStride}`）和多维 slice（`SliceInfo`，最多 8 维），**不支持传入 indices 数组做任意地址 scatter/gather**。这直接决定了 miss copy 的实现形态（见 §2.7.3 Phase 2）。

#### 2.7.2 与 GPU kernel 的对应关系

| GPU swap-in kernel（warp 内） | NPU residency kernel（分 pipe） |
|------|------|
| 32 lanes 并行 probe hash table | Vector Unit 批量比较 + Scalar 收集结果 |
| lane 各自 `ld.global.nc` scatter read host | Scalar 循环派发 + MTE2 逐 token DataCopy |
| warp 内隐式同步 | SetFlag/WaitFlag 跨 queue 显式同步 |
| 一个 warp 同时做判定 + 搬运 | 判定（Vector/Scalar）与搬运（MTE）分离到不同 pipe，靠 double buffering overlap |

#### 2.7.3 算法伪代码（分 pipe）

> **实证锚点**：下面的三阶段写法不是凭空设计——外部包里现成的 AscendC 算子 [`cache_loc_assign_kernel.cpp`](../sgl-kernel-npu/csrc/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp)（§2.2.1 (5)）已经示范了同款骨架:`GetBlockIdx()` 做 per-core tiling、`TPipe`+`TBuf`/`TQue` 手工管 UB、`DataCopy`/`DataCopyPad` 走 MTE、`SetFlag/WaitFlag<HardEvent::MTE2_V / V_MTE3>` 卡流水级。新 kernel 可直接以它为模板。它也用 `for j: GetValue/SetValue` 的逐元素标量循环，**实测印证了 Phase 1 必须用 Vector 批量比较、Phase 2 必须走 MTE 整块搬运**——任何落到 Scalar 逐元素的路径都是性能陷阱。

**Phase 1 — Hit/Miss 判定（Vector 主导）**

```
# Scalar: 加载跟踪表和请求
DataCopy(UB.device_buffer_tokens ← GM.device_buffer_tokens)  # MTE2, [buffer_size]
DataCopy(UB.top_k_tokens ← GM.top_k_tokens)                  # MTE2, [K]

# Vector: 批量判定每个 top_k token 是否已在 hot buffer
for each top_k token t (向量化，非标量循环):
    Compare(UB.hit_mask ← UB.device_buffer_tokens, t)        # Vector，批量比较 buffer_size 个 slot
    hit_slot = first set bit in hit_mask                     # 命中则记录 physical slot

# Scalar: 收集结果
miss_list   = [t for t in top_k if not hit]
evict_list  = LRU 选出 miss_list.size 个可驱逐 slot
更新 lru_state（hit 的刷新时间戳，evict 的标记）
```

**Phase 2 — Miss Copy（Scalar 派发 + MTE pipeline）**

```
# 无法一次 scatter，逐 miss 发射，double buffer overlap
InitBuffer(copy_que, 2, token_size)   # double buffering

for i, (miss_tok, evict_slot) in enumerate(zip(miss_list, evict_list)):
    src = host_kv_devPtr + miss_tok.pos * token_stride       # Scalar 算地址
    dst = hot_buffer     + evict_slot   * token_stride
    ub  = copy_que[i % 2]
    DataCopy(ub  ← src, token_size)    # MTE2: Host GM → UB
    DataCopy(dst ← ub,  token_size)    # MTE3: UB → hot buffer GM
    更新 device_buffer_tokens[evict_slot] = miss_tok
# 第 i 个 miss 的 MTE3 写出与第 i+1 个的 MTE2 读入并行
```

**Phase 3 — 输出**

```
for each top_k token t:
    top_k_device_locs[t] = hit_slot if hit else assigned_evict_slot
DataCopy(GM.top_k_device_locs ← UB.top_k_device_locs)        # MTE3
```

##### 2.7.3.1 三个"死穴"原语的 SIMD-native 替代（均有 sgl-kernel-npu 代码实证）

GPU 用的 warp 原语（hash 表 probe、ballot、shfl）在 NPU 上没有对应，照搬会退化成 Scalar 逐元素循环（§2.2.1 (5) 已实测此陷阱）。但它们要达成的**目的**都有 SIMD-native 的替代，且**不是纸面推演——下列 API 都能在已拉取的 `sgl-kernel-npu` 生产算子里找到真实在用的调用**：

| 死穴（GPU 原语） | SIMD-native 替代 | 代码实证（sgl-kernel-npu） |
|--|--|--|
| **hit 判定**：hash 表随机 probe（warp 内免费，NPU 上是 scatter 死穴） | **排序 + 向量化求交**：驻留表 token-id 保持有序、top_k 排序，做有序集合 merge intersection（纯规则比较，零 scatter） | `lightning_indexer` 有完整向量化全排序 [`lightning_indexer_vector.h:183-257`](../sgl-kernel-npu/csrc/lightning_indexer/op_kernel/lightning_indexer_vector.h#L183)：`AscendC::Sort32` + `MrgSort` 多级归并，注释明确支持 **`[128,256,384,512,1024,2048]` 个元素的 value+index 配对排序**——正好是 top_k≈2048 规模 |
| **miss 收集 / scatter**：warp 压缩出紧凑 miss_list | **`CompareScalar` 产掩码 + `GatherMask` 向量化收集**（消灭逐元素 SetValue） | DeepEP MoE dispatch 大量在用：`CompareScalar(..., CMPMODE::EQ/LT, ...)` + `GatherMask`（[`moe_distribute_dispatch_v2_single.h:728`](../sgl-kernel-npu/csrc/deepep/ops2/op_kernel/moe_distribute_dispatch_v2_single.h#L728)） |
| **LRU 驱逐**：ballot/shfl 并行选最久未用槽 | **时间戳 + 向量化 `argmin`/归约** | `WholeReduceMax`/`BlockReduceMax`/`ReduceMax` 在 `mla_preprocess`、DeepEP 多处在用（[`simd.h:82`](../sgl-kernel-npu/csrc/mla_preprocess/op_kernel/kernel/simd.h#L82)） |

**这把"判定平面"整体从依赖 host 直读中解耦出来**：上述四步（排序求交判定 + GatherMask 收集 miss_list + 归约选驱逐槽 + double-buffer DMA 搬运）**全部走 device 上 proven 的 SIMD API、且完全不依赖 §2.2.2 的 host 直读能力**。含义见 §2.7.5 末尾的架构判断。

> **粗化粒度（SIMD-native 的备选优化）**：GPU 是 token/页级驻留，scatter 压力大但 warp 不在乎；SIMD 在乎。可把驻留粒度从单 token 粗化到"块"（一次换入连续一段），用一点缓存精度换**判定与搬运都变规则**（求交更整齐、DMA 更连续、几乎消灭 scatter）。这个 trade-off 在 GPU 上不划算、在 NPU 上要重新算账。落地形态正是路线 A′（PA_BSND + 重映射 block_table，页粒度驻留，§2.3.2）。

#### 2.7.4 UB 内存分配

| buffer | 大小（buffer_size=4096, K=2048, BF16） | 说明 |
|--------|------|------|
| device_buffer_tokens | 4096 × 4B = 16 KB | 跟踪表（token id, int32） |
| top_k_tokens | 2048 × 4B = 8 KB | 本步请求 |
| hit_mask / 中间向量 | ~数 KB | Vector 比较中间结果 |
| copy double buffer | 2 × token_size（token_stride ≈ 1152B）≈ 2.3 KB | miss copy 暂存 |

总占用远小于 UB 容量（192 KB+），double buffer 可放大到更多级以加深 pipeline。

#### 2.7.5 性能分析

| 环节 | NPU | vs GPU |
|------|-----|--------|
| hit 判定 | Vector 批量比较 buffer_size 个 slot，向量化 | GPU warp 并行 probe。**Vector 是必须而非可选**——官方明确 Scalar 计算弱、应 minimize；纯 Scalar 线性扫描 4096 slot 是瓶颈 |
| miss copy | 逐 token DataCopy + double buffer overlap | GPU PTX streaming scatter。NPU 搬连续整 token（DMA 强项），**效率可能持平或更优**，但受限于"无法一次 scatter 所有 miss" |
| 地址计算 | Scalar 逐 miss 算 src/dst | GPU lane 内算。miss 数多时 Scalar 派发开销累积 |

结论：miss copy 不是瓶颈（DMA 搬连续块是 NPU 强项），**hit 判定必须走 Vector 向量化**，Scalar 仅做派发和收集。

**架构判断（结合 §2.2.2 与 §2.7.3.1）**：判定平面四步全走 device 上 proven 的 SIMD API、不依赖 host 直读，因此它本身就是一个**完整的路径②实现**——判定全程在 device kernel 内、只把紧凑 miss_list 交给 host 发 staged DMA。**这条路不必等 P0b 就能开写。** P0b 因此从"成败门控"降为"优化"：通过则把搬运也吸进 kernel（路径①，Graph 更友好）；不通过则判定平面 + host staged DMA 照跑（路径②，多一趟往返、Graph 需分段）。两条路都有 proven 退路。

#### 2.7.6 参考文档

1. [CANN 8.5 Basic Architecture（AI Core 硬件架构）](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0007.html)
2. [CANN 8.5 Abstract Hardware Architecture（编程模型抽象）](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
3. [CANN 8.0 Control Units（指令队列与同步）](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
4. [Double Buffering Best Practice](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0033.html)
5. [TPipe/TQue Programming Paradigm](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00033.html)
6. [Basic DataCopy API（strided copy）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0103.html)
7. [Slice DataCopy API（多维 slice）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0105.html)

### 2.8 NPU Stream / Event 同步模型

> **导读**：同 01 §2.6 的思路——搬运与计算分到不同流水线并行、用 event 当红绿灯保证先后。必须守住的安全规则只有一条：**仓库还没存好这一页的副本前，书桌上这一格绝不能被新字覆写**，否则该页永久丢失。下面的时序图与 event 等待都为守住这一条。

#### 2.8.1 CANN stream/event 对应

| CUDA | CANN | 用途 |
|------|------|------|
| `cudaStream_t` | `aclrtStream` | 异步执行队列 |
| `cudaEvent_t` | `aclrtEvent` | 跨 stream 同步点 |
| `cudaStreamCreate` | `aclrtCreateStream` | 创建 stream |
| `cudaEventCreate` | `aclrtCreateEvent` | 创建 event |
| `cudaStreamWaitEvent` | `aclrtStreamWaitEvent` | stream 等待 event（非阻塞 host，stream 内排队等待） |
| `cudaEventRecord` | `aclrtRecordEvent` | 在 stream 上记录 event |
| `cudaEventSynchronize` | `aclrtSynchronizeEvent` | host 阻塞等待 event 完成 |
| `cudaEventQuery` | `aclrtQueryEventStatus` | host 非阻塞查询 event 状态 |
| `cudaStreamSynchronize` | `aclrtSynchronizeStream` | host 阻塞等待 stream 全部完成 |
| `cudaMemcpyAsync` | `aclrtMemcpyAsync`（`ACL_MEMCPY_DEVICE_TO_HOST`） | 异步 D2H / H2D copy |

API 名称与语义依据 AscendCL acl API (C) 官方文档核实（链接见 §2.8.4）。

HiSparse 用两条 stream：`main_stream`（forward decode）和 `backup_stream`（D2H 备份上一步 token）。

#### 2.8.2 时序（对标 doc01 §2.6 GPU 时序）

```
main_stream    │ write KV ──┬─ indexer ─ residency kernel ─ SFA ─ ... (step N)
               │            │
               │     record main_evt
               │            │
backup_stream  │   wait main_evt ─ D2H copy(prev token) ─ record backup_evt
               │                                              │
main_stream    │ ... (step N+1) wait backup_evt ─ write KV ───┘ (覆写 prev slot 前必须等 backup 完成)
```

关键约束：**main stream 覆写某 slot 前，该 slot 上一 token 的 D2H 备份必须已完成**——否则全量 KV 在 Host 上会丢失尚未备份的 token。靠 `backup_evt` 在 step N+1 的 KV 写入前对 main_stream 设 `aclrtStreamWaitEvent` 保证。

#### 2.8.3 与 GPU 模型的差异

| 维度 | GPU | NPU |
|------|-----|-----|
| Graph capture | CUDA Graph 天然支持含 host read 的 kernel | **SFA 本身官方明确支持图模式**（`return_softmax_lse=False` 时，§2.3.2）；但**自研 residency kernel 若含 registered-host devPtr 读**（路径①），其可 capture 性 **待 P4 验证**（路径②的判定 kernel 不含 host read，进图无此风险） |
| host read 在 stream 内行为 | UVA + PTX streaming，无显式同步 | devPtr DataCopy 走 MTE，是否需额外 barrier 待 P0b 验证 |
| 多 stream 调度 | SM 级抢占 | AI Core 调度粒度待确认，backup_stream 可能与 main_stream 争抢 MTE |

P0 探针（§2.6）需顺带验证 backup_stream 的 D2H 与 main_stream 的 residency kernel 在 MTE 上是否互相阻塞。

#### 2.8.4 参考文档

1. [AscendCL Synchronous Wait（aclrtSynchronizeStream / aclrtStreamWaitEvent / aclrtSynchronizeEvent）](https://www.hiascend.com/document/detail/en/canncommercial/850/appdevg/acldevg/aclcppdevg_000013.html)
2. [aclrtCreateStream（Stream Management）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0065.html)
3. [aclrtRecordEvent（Event Management）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0083.html)
4. [aclrtMemcpyAsync（Memory Management，含 `ACL_MEMCPY_DEVICE_TO_HOST`）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0106.html)
5. [acl API (C) API List（Atlas A2，含 event/stream 完整列表）](https://www.hiascend.com/document/detail/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_0014.html)

### 2.9 验收标准

1. **正确性**：HiSparse 输出 vs full-resident 输出的 logits max diff < 1e-3
2. **资源**：NPU KV 显存 ∝ O(top_k) 而非 O(max_seq_len)
3. **性能**：拆出 indexer / residency kernel / SFA / backup 四段耗时

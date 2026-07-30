# 决策：每个扩展 seam 都必须提供稳定 C ABI，支持动态库加载

**日期**：2026-07-30
**决策人**：@justinchuby（owner）
**来源**：#524 Q1（契约审计提出的元决策）
**状态**：已拍板，standing directive

## 决策内容

第三方出树扩展的目标形态是：**每个扩展 seam 都要有稳定 C ABI，全面支持 `.dll` / `.so` 动态加载。**

不接受「只提供编译期 Rust trait、要求第三方链接本 workspace」的形态。

## 适用范围

所有扩展点，包括但不限于：

- Execution Provider（native EP，不只是现有的 legacy plugin EP）
- `DeviceAllocator`（自带内存管理）
- `MemoryPlanner`（激活/workspace 规划）
- `KvCacheStore` 与 `KvCacheConnector`
- `Sampler` / `LogitProcessor` / `SpeculativeProposer`
- `SchedulingPolicy`
- `OptimizationPass`（fusion / 图优化）
- `Kernel`（为已有 EP 增加或替换单个 kernel）
- `PlacementCostModel`、`WeightEvictionPolicy`、`ReclaimPolicy`
- `Communicator`（跨设备传输）

## 直接推论

1. **Rust trait 仍然需要**，但它是**进程内实现层**，不是边界。每个 seam 需要 trait + C vtable 双向 shim（host→plugin 与 plugin→host）。
2. **ABI 基座成为所有 seam 的前置**：版本协商、错误传播、panic fencing、跨边界所有权约定必须先统一，否则每个 seam 各造一套。
3. **稳定性策略从 P2 升为 P0**（原 #512）。既然对外承诺 ABI，就必须同时给出稳定性等级与版本化机制。
4. **现有插件 EP 的 C ABI 是唯一已验证的样板**（`crates/onnx-runtime-ep-api/src/abi/runtime.rs:33-132`，含 `ort_version_supported` 版本协商，`registry.rs:220-226` 的 `libloading` + `CreateEpFactories` 加载）。应将其提炼为所有 seam 复用的模式。
5. **RULES.md §1 的 FFI 要求成为硬约束**：C ABI 调用必须返回机器可解析的错误码 + 可取回的富文本消息；绝不丢弃 Rust cause；绝不跨 FFI unwind。
6. **DLPack 的定位大概率随之确定**（#524 Q3）：既然边界是 C ABI，DLPack 作为跨实现零拷贝 tensor 交换的现成 C ABI 标准，是自然选择。待 owner 在 #524 确认。
7. **Q2 的紧迫性上升**：EP ABI 是对齐上游 ORT plugin ABI 还是定义 nxrt 自有 ABI —— 这个选择现在会成为其余所有 seam ABI 的模板。

## 影响的 issue

- 新增：插件 ABI 基座（版本协商 / 错误传播 / panic fence / 所有权约定 / 一致性测试）
- 升级：#512 稳定性策略 P2 → P0
- 形态变更：#506、#508、#509、#510、#511、#513、#514、#515、#516、#517、#518、#519、#520 —— 每个都从「加 Rust trait」变为「Rust trait + C ABI vtable + 版本化」

## 备注

此决策显著提高了每个 seam 的工作量，但符合项目「第三方可针对性优化、无需 fork」的核心目标。建议按 seam 分批冻结 ABI，而非一次性全部设计。

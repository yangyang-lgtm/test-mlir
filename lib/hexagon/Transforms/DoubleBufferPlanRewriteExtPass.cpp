//===- DoubleBufferPlanRewriteExtPass.cpp -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h" // 本 pass 直接识别并生成 memref_ext 的 load/store/DMA 相关操作。
#include "hexagon/Transforms/Passes.h"
#include "triton-shared/Conversion/MemorySpaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // 需要处理 tt.addptr 这类 Triton 指针计算，追溯真实内存来源。

#include "mlir/Analysis/AliasAnalysis.h" // 需要别名分析来判断 tile 根内存是否可能互相影响。
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h" // 需要读取操作的内存副作用，才能建立安全的数据依赖。
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERPLANREWRITEEXT
#include "hexagon/Transforms/Passes.h.inc"

namespace { // 这些工具函数只服务本 pass，不暴露到链接符号表。

constexpr StringLiteral kPlanIdAttr = "db_plan_id"; // 给新生成的 prologue/loop 打同一个 id，方便后续 pass 识别同一计划。
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue"; // 标记首轮预取区域，后续 lower 能区分它和 steady-state 预取。
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch"; // 标记循环内下一轮预取区域，帮助 DMA lowering 做角色区分。
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute"; // 标记克隆出的计算，避免后续 pass 把它误当成原始传输。
constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role"; // 传输角色需要保留下来，后续 DMA lowering 才知道是 prefetch 还是 store。

enum TileAccessKind : uint8_t { // 用位标志表达一个节点对每个 tile 的读写关系，便于构建依赖。
  NoTileAccess = 0, // 没有访问的 tile 不参与当前节点依赖，减少不必要约束。
  TileRead = 1, // 读标志用于建立对最近写的 RAW 依赖。
  TileWrite = 2, // 写标志用于建立 WAW 和对历史读的 WAR 依赖。
};

struct TileNode { // 把 loop body 操作抽象成 tile 数据流节点，后面按节点重写。
  Operation *op = nullptr;
  SmallVector<uint8_t> accesses;
  SmallVector<unsigned> predecessors;
  bool isLoad = false; // load 节点会在重写时变成 prologue/next-iteration prefetch。
  bool isCompute = false; // compute 节点会被克隆到 double-buffer loop 的等待之后。
  bool isStore = false; // store 节点会被克隆到当前轮计算之后执行写回。
};

struct TileInfo { // 每个 tile 记录根内存和可选 load，决定是否需要双缓冲。
  Value root; // 根内存是原 loop 中所有视图映射到 ping/pong buffer 的键。
  Operation *load = nullptr; // load 存在说明该 tile 每轮从全局内存预取，需要双份 buffer。
};

struct Plan { // Plan 汇总分析结果，rewritePlan 只消费这个结构做模板化重写。
  scf::ForOp loop; // 保存要替换的原始循环，重写结束后会删除它。
  SmallVector<TileInfo> tiles;
  SmallVector<TileNode> nodes;
  SmallVector<Operation *> steadyOps; // 按 body 顺序需要克隆进 steady-state 的算子（含 compute/store/scratch alloc）。
  bool isReduction = false; // reduce loop 可以没有显式 store，分类条件需要放宽。
};

enum class TransferKind { Load, Store };

struct TransferLike {
  TransferKind kind;
  Operation *op;
  Value source;
  Value target;

  bool isLoad() const { return kind == TransferKind::Load; }
  bool isStore() const { return kind == TransferKind::Store; }
  Value getSource() const { return source; }
  Value getTarget() const { return target; }
  Value getTile() const { return isLoad() ? target : source; }
};

std::optional<TransferLike> getTransferLike(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<TransferLike>>(op)
      .Case<memref_ext::LoadExOp>([&](auto load) {
        return TransferLike{TransferKind::Load, op, load.getPtr(),
                            load.getTarget()};
      })
      .Case<memref_ext::BlockDescriptorLoadOp,
            memref_ext::BlockDescLoadOp>([&](auto load) {
        return TransferLike{TransferKind::Load, op, load.getDesc(),
                            load.getTarget()};
      })
      .Case<memref_ext::StoreExOp>([&](auto store) {
        return TransferLike{TransferKind::Store, op, store.getValue(),
                            store.getPtr()};
      })
      .Case<memref_ext::BlockDescriptorStoreOp,
            memref_ext::BlockDescStoreOp>([&](auto store) {
        return TransferLike{TransferKind::Store, op, store.getValue(),
                            store.getDesc()};
      })
      .Default([](Operation *) -> std::optional<TransferLike> {
        return std::nullopt;
      });
}

Value findMemoryRoot(Value value) { // 规约 memref view/cast/addptr，确保同一 tile 的不同 SSA 值能匹配。
  while (value) { // 沿着等价内存值链向上追溯，直到找到能用于依赖判断的根。
    if (auto alloc = value.getDefiningOp<memref::AllocOp>()) // 只有落到本地 alloc，后续才能为同一 tile 分配 ping/pong 并重写所有视图使用。
      return alloc.getMemref();
    Operation *def = value.getDefiningOp();
    if (!def) // 计划重写只支持本地 alloc tile；block 参数或外部值不能分配对应 ping/pong 根。
      return {}; // 找不到可信根内存时返回空值，让调用方走保守路径。
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) { // 这些 op 只是改变视图或类型，不应切断同一块内存的依赖关系。
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) { // 这些 op 只是改变视图或类型，不应切断同一块内存的依赖关系。
      value = cast.getSource();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) { // 这些 op 只是改变视图或类型，不应切断同一块内存的依赖关系。
      value = reinterpret.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) { // 这些 op 只是改变视图或类型，不应切断同一块内存的依赖关系。
      value = view.getSource();
      continue;
    }
    if (auto addPtr = dyn_cast<triton::AddPtrOp>(def)) { // tt.addptr 只做指针偏移，依赖判断要回到原始指针。
      value = addPtr.getPtr();
      continue;
    }
    return {}; // 找不到可信根内存时返回空值，让调用方走保守路径。
  }
  return {}; // 找不到可信根内存时返回空值，让调用方走保守路径。
}

bool isAvailableBefore(Value tile, scf::ForOp loop, Operation *op) { // tile 根必须在克隆位置可用，否则无法映射到外提的 ping/pong buffer。
  Operation *def = tile.getDefiningOp();
  if (!def) // 没有定义 op 的 tile 不能证明在重写位置可用，因此不能纳入模板。
    return false;
  Block *body = loop.getBody();
  if (def->getBlock() == body) // tile 若在 loop 内定义，必须支配 load/store；否则克隆 prologue 或 compute 时会引用尚未生成的值。
    return def->isBeforeInBlock(op);
  return def->getBlock() == loop->getBlock() && def->isBeforeInBlock(loop);
}

StringRef accessName(uint8_t access) { // 把位标志打印成读写名称，辅助检查数据流图。
  if (access == (TileRead | TileWrite)) // 同一节点既消费旧值又产生新值，依赖图必须同时解释 RAW/WAR/WAW 来源。
    return "READ_WRITE";
  if (access == TileRead) // 纯读只需要等最近写完成，不会更新 tile 的版本。
    return "READ";
  if (access == TileWrite) // 纯写会生成 tile 新版本，后续读和写都必须以它作为顺序边界。
    return "WRITE";
  return "NONE";
}

struct ExtTileDataFlowAnalysis {
  ExtTileDataFlowAnalysis(scf::ForOp loop, AliasAnalysis &aliasAnalysis)
      : loop(loop), aliasAnalysis(aliasAnalysis) {}

  bool run(Plan &plan) { // 只有 tile 收集、依赖建图和分类都成功，才允许进入结构化重写。
    plan.loop = loop; // 保存要替换的原始循环，重写结束后会删除它。
    if (!collectTiles(plan) || !buildNodes(plan) || !classify(plan)) // tile 收集、依赖建图、角色分类任一失败，都说明原 loop 不匹配固定 prologue+prefetch+compute+store 模板。
      return false;
    print(plan, llvm::outs());
    return true; // 分析或重写条件已经满足，可以继续后续 pipeline。
  }

  bool collectTiles(Plan &plan) { // 先找到所有参与 load/store_ext 的本地 tile，重写时才能统一替换成新 buffer。
    for (Operation &op : loop.getBody()->without_terminator()) { // 只分析 loop body 的实际操作，terminator 不参与 tile 数据流。
      if (std::optional<TransferLike> transfer = getTransferLike(&op)) {
        Value root = findMemoryRoot(transfer->getTile()); // 根内存是原 loop 中所有视图映射到 ping/pong buffer 的键。
        if (!root || !isAvailableBefore(root, loop, transfer->op)) // transfer 访问的 tile 必须支配该操作，否则克隆后可能引用未定义值。
          return false;
        if (!addTile(root, transfer->isLoad() ? transfer->op : nullptr,
                     plan)) // load 记录预取模板，store-only tile 只记录 root 供 currentMapping 替换。
          return false;
        continue;
      }
    }
    return !plan.tiles.empty();
  }

  bool addTile(Value root, Operation *load, Plan &plan) { // 根内存是原 loop 中所有视图映射到 ping/pong buffer 的键。
    for (TileInfo &tile : plan.tiles) {
      if (tile.root != root) // 根不同就代表不同 tile，不能把它们合并到同一个 ping/pong buffer 槽。
        continue;
      if (load && tile.load) // 同一 tile 已经记录过预取 load，再遇到一个 load 会让“首轮/下一轮只克隆一次”这个模板失效。
        return false;
      if (load) // 当前 root 已存在且这次来自 load_ex，记录它作为该 tile 的唯一预取模板。
        tile.load = load;
      return true;
    }
    plan.tiles.push_back(TileInfo{root, load});
    return true;
  }

  std::optional<unsigned> getTileIndex(Value value, const Plan &plan) { // 把任意 memref SSA 值映射回 tile 编号，依赖建图都基于编号进行。
    value = findMemoryRoot(value); // 规约 memref view/cast/addptr，确保同一 tile 的不同 SSA 值能匹配。
    if (!value) // 追不到根内存就无法和 plan.tiles 做等价/别名匹配，不能把它归到任何 tileIndex。
      return std::nullopt;
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      if (value == tile.root || aliasAnalysis.alias(value, tile.root)) // SSA 值等于或可能别名 tile root 时，重写必须把它归入同一个 buffer 槽。
        return index;
    return std::nullopt;
  }

  bool addEffectAccesses(Operation *op, TileNode &node, Plan &plan) { // 优先用精确内存效果判断 compute 是否读写 tile。
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp) // op 不声明内存副作用接口时，精确路径无法判断它访问哪些 tile，需要调用方走保守 ModRef。
      return false; // 当前 IR 形态无法证明安全，放弃这次优化以保持语义。
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects); // 读取操作声明的副作用，避免把无关操作当成 tile 依赖。
    for (const MemoryEffects::EffectInstance &effect : effects) {
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
      if (!isRead && !isWrite) // allocate/free 等非读写效果不表示 tile 内容流动，不应产生数据依赖边。
        continue;
      Value value = effect.getValue();
      if (!value || !isa<BaseMemRefType>(value.getType())) // 读写效果必须绑定到具体 memref；否则无法判断是否命中计划 tile，只能放弃精确分析。
        return false; // 当前 IR 形态无法证明安全，放弃这次优化以保持语义。
      if (auto tile = getTileIndex(value, plan)) { // effect 的 memref 命中计划 tile 时，才会影响重写后 buffer 的读写顺序。
        if (isRead) // 读效果说明该 op 会消费当前 tile 内容，重写后必须等待最近写者。
          node.accesses[*tile] |= TileRead; // 读标志用于建立对最近写的 RAW 依赖。
        if (isWrite) // 写效果说明该 op 会覆盖 tile 内容，重写后必须保留写后写和读后写顺序。
          node.accesses[*tile] |= TileWrite; // 写标志用于建立 WAW 和对历史读的 WAR 依赖。
      }
    }
    return true;
  }

  void addConservativeAccesses(Operation *op, TileNode &node, Plan &plan) { // 无法精确解析效果时用 ModRef 保守建依赖，宁可少优化也不改语义。
    for (auto [index, tile] : llvm::enumerate(plan.tiles)) {
      ModRefResult modRef = aliasAnalysis.getModRef(op, tile.root); // 询问操作是否可能读或写 tile 根内存，用于未知效果的保守处理。
      if (modRef.isRef()) // ModRef 认为 op 可能读取该 tile，就按读处理以防 compute 被提前到生产者之前。
        node.accesses[index] |= TileRead; // 读标志用于建立对最近写的 RAW 依赖。
      if (modRef.isMod()) // ModRef 认为 op 可能修改该 tile，就按写处理以防覆盖顺序在模板重写后变化。
        node.accesses[index] |= TileWrite; // 写标志用于建立 WAW 和对历史读的 WAR 依赖。
    }
  }

  bool buildNodes(Plan &plan) { // 把 loop body 的 tile 访问转成依赖图，保证重写后的顺序等价。
    SmallVector<std::optional<unsigned>> lastWrite(plan.tiles.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(plan.tiles.size());
    for (Operation &op : loop.getBody()->without_terminator()) { // 只分析 loop body 的实际操作，terminator 不参与 tile 数据流。
      TileNode node;
      node.op = &op;
      node.accesses.assign(plan.tiles.size(), NoTileAccess); // 没有访问的 tile 不参与当前节点依赖，减少不必要约束。
      if (std::optional<TransferLike> transfer = getTransferLike(&op)) {
        if (transfer->isLoad()) {
          node.isLoad = true; // load 节点会在重写时变成 prologue/next-iteration prefetch。
          if (auto target = getTileIndex(transfer->getTarget(), plan)) // load 写入计划 tile，代表生产本地 tile。
            node.accesses[*target] |= TileWrite; // 写标志用于建立 WAW 和对历史读的 WAR 依赖。
        } else {
          node.isStore = true; // store 节点会被克隆到当前轮计算之后执行写回。
          if (auto value = getTileIndex(transfer->getSource(), plan)) // store 读取计划 tile，代表消费本地 tile 后写回。
            node.accesses[*value] |= TileRead; // 读标志用于建立对最近写的 RAW 依赖。
        }
      } else if (!isa<memref::AllocOp, memref::DeallocOp, memref::SubViewOp,
                    memref::ViewOp, memref::CastOp,
                    memref::ReinterpretCastOp>(&op) &&
                 !isMemoryEffectFree(&op)) {
        if (!addEffectAccesses(&op, node, plan)) // 精确 effect 不能覆盖这个 op 时，退回 alias ModRef，宁可多建边也不能漏掉 tile 访问。
          addConservativeAccesses(&op, node, plan); // 无法精确解析效果时用 ModRef 保守建依赖，宁可少优化也不改语义。
      } else if (!isMemoryEffectFree(&op) &&
                 !addEffectAccesses(&op, node, plan)) { // 优先用精确内存效果判断 compute 是否读写 tile。
        addConservativeAccesses(&op, node, plan); // 无法精确解析效果时用 ModRef 保守建依赖，宁可少优化也不改语义。
      }
      if (!llvm::any_of(node.accesses, // 该 op 对所有计划 tile 都是 NoTileAccess，模板重写不需要克隆或移动它。
                        [](uint8_t access) { return access != NoTileAccess; })) // 没有访问的 tile 不参与当前节点依赖，减少不必要约束。
        continue;
      unsigned nodeIndex = plan.nodes.size();
      llvm::SmallDenseSet<unsigned> predecessors;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) { // 逐个 tile 读写关系转成 RAW/WAR/WAW 依赖。
        bool reads = access & TileRead; // 读标志用于建立对最近写的 RAW 依赖。
        bool writes = access & TileWrite; // 写标志用于建立 WAW 和对历史读的 WAR 依赖。
        if (reads && lastWrite[tileIndex]) // 当前读会消费最近写出的 tile 版本，模板重排后也必须保持这条 RAW 顺序。
          predecessors.insert(*lastWrite[tileIndex]);
        if (writes) { // 写操作会结束此前读者的开放区间，并成为新的最近写。
          if (lastWrite[tileIndex]) // 如果已有最近写者，当前写必须依赖它，否则两个写在重写后可能交换最终结果。
            predecessors.insert(*lastWrite[tileIndex]);
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
          if (reads) // 读写节点也要作为读者登记，避免后续写被提前到它读旧值之前。
            readsSinceWrite[tileIndex].push_back(nodeIndex);
        } else if (reads) {
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors); // 稳定排序让调试输出和重写决策不依赖哈希/集合顺序。
      plan.nodes.push_back(std::move(node));
    }
    return true;
  }

  bool isTileRootAlloc(Operation *op, const Plan &plan) { // tile 的 alloc 会被 ping/pong buffer 取代，不能再克隆一份。
    auto alloc = dyn_cast<memref::AllocOp>(op);
    if (!alloc)
      return false;
    for (const TileInfo &tile : plan.tiles)
      if (tile.root == alloc.getMemref()) // 结果正好是某个 tile 根内存时，交给 currentMapping 映射到 buffer。
        return true;
    return false;
  }

  bool classify(Plan &plan) { // 收集需要克隆进 steady-state loop 的 body 算子（按程序顺序）。
    Block *body = loop.getBody();
    Operation *terminator = body->getTerminator();

    auto isExcluded = [&](Operation *op) { // load 由预取克隆，tile alloc 由 buffer 取代，二者都不进 steady body。
      if (std::optional<TransferLike> transfer = getTransferLike(op))
        if (transfer->isLoad())
          return true;
      return isTileRootAlloc(op, plan);
    };

    DenseSet<Operation *> needed; // steady body 需要的算子集合，闭包收集避免漏掉累加器链。
    SmallVector<Operation *> worklist;
    auto markNeeded = [&](Operation *op) {
      if (!op || op->getBlock() != body || op == terminator) // 只克隆 loop body 内、非 terminator 的定义。
        return;
      if (isExcluded(op)) // 排除项由 mapping 覆盖，不需要显式克隆。
        return;
      if (needed.insert(op).second)
        worklist.push_back(op);
    };

    // 种子：所有写内存/带副作用的算子（linalg.map、memref.copy、scratch alloc、store 等）。
    // 累加器/scratch 数据通过 memref 就地写入流动，只有把这些副作用算子全部纳入，
    // 语义才不会因为“不碰双缓冲 tile”而丢失。
    for (Operation &op : body->without_terminator()) {
      if (isExcluded(&op))
        continue;
      if (!isMemoryEffectFree(&op) || getTransferLike(&op))
        markNeeded(&op);
    }
    // yield 出去的循环携带值也必须能被算出。
    for (Value operand : terminator->getOperands())
      markNeeded(operand.getDefiningOp());
    // 沿 SSA 操作数向上闭包，把上述算子依赖的纯计算也补齐。
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (Value operand : op->getOperands())
        markNeeded(operand.getDefiningOp());
    }

    bool hasStore = false;
    for (Operation &op : body->without_terminator()) { // 保持原 body 顺序，确保副作用次序不变。
      if (!needed.contains(&op))
        continue;
      if (std::optional<TransferLike> transfer = getTransferLike(&op))
        if (transfer->isStore()) {
          if (!getTileIndex(transfer->getSource(), plan)) // store 源必须是计划 tile，否则 currentMapping 无法替换数据来源。
            return false;
          hasStore = true;
        }
      plan.steadyOps.push_back(&op);
    }

    bool hasLoadTile = // 至少要有一个带 load 的 tile，双缓冲预取才有意义。
        llvm::any_of(plan.tiles, [](const TileInfo &t) { return t.load; });
    return !plan.steadyOps.empty() && hasLoadTile &&
           (plan.isReduction || hasStore); // 非 reduce 还必须有 store，否则重写后没有明确写回阶段。
  }

  void print(const Plan &plan, raw_ostream &os) { // 调试输出让重写前的数据流形态可见，方便排查误分类。
    os << "\n=== HexagonDoubleBufferPlanRewriteExt dataflow ===\n";
    os << "loop: ";
    loop->print(os, OpPrintingFlags().skipRegions());
    os << "\n";
    os << "nodes:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes)) {
      os << "  [" << nodeIndex << "] ";
      if (node.isLoad) // 打印 load 角色，便于核对它是否只出现在预取阶段。
        os << "LOAD_EXT";
      else if (node.isStore) // 打印 store 角色，便于核对它是否只出现在写回阶段。
        os << "STORE_EXT";
      else
        os << "COMPUTE";
      os << " accesses={";
      bool first = true;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) { // 逐个 tile 读写关系转成 RAW/WAR/WAW 依赖。
        if (access == NoTileAccess) // 只输出实际参与依赖的 tile，调试时能聚焦重写依据。
          continue;
        if (!first) // 已经输出过一个 tile 访问时加分隔符，避免多个访问项黏在一起。
          os << ", ";
        first = false;
        os << "tile#" << tileIndex << ":" << accessName(access); // 把位标志打印成读写名称，辅助检查数据流图。
      }
      os << "}\n      op: ";
      node.op->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }
    os << "dataflow:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes))
      for (unsigned predecessor : node.predecessors)
        os << "  [" << predecessor << "] -> [" << nodeIndex << "]\n";
    os << "=== End HexagonDoubleBufferPlanRewriteExt dataflow ===\n";
  }

  scf::ForOp loop; // 保存要替换的原始循环，重写结束后会删除它。
  AliasAnalysis &aliasAnalysis;
};

bool isDefinedBefore(Operation *op, Block *block, Operation *limit) { // 只克隆当前操作之前的本地定义，外部值或较晚定义不能提前复制。
  return op && op->getBlock() == block && op->isBeforeInBlock(limit);
}

Value cloneSlice(Value value, IRRewriter &rewriter, IRMapping &mapping, // 克隆操作前先克隆它依赖的局部定义，保证新位置的 operand 可用。
                 Block *sourceBlock, Operation *limit) {
  if (Value mapped = mapping.lookupOrNull(value)) // value 若已映射，说明它是 iv、tile root 或已克隆定义，必须复用映射值以保持 SSA 一致。
    return mapped;
  Operation *def = value.getDefiningOp();
  if (!isDefinedBefore(def, sourceBlock, limit)) // 只克隆当前 op 之前的同 block 定义；外部值已支配，新定义或后置定义不能被提前复制。
    return value;
  for (Value operand : def->getOperands())
    cloneSlice(operand, rewriter, mapping, sourceBlock, limit); // 克隆操作前先克隆它依赖的局部定义，保证新位置的 operand 可用。
  Operation *cloned = def->clone(mapping); // 克隆依赖定义，让被克隆操作在新区域内自洽。
  rewriter.insert(cloned);
  mapping.map(def->getResults(), cloned->getResults()); // 建立原 SSA 值到克隆/替换值的映射，保证克隆操作使用新 buffer。
  return mapping.lookup(value);
}

Operation *cloneMappedOp(Operation *op, IRRewriter &rewriter, // 按当前轮映射克隆 compute/store/load，保持原操作语义但替换 tile。
                         IRMapping &mapping, Block *sourceBlock,
                         bool markCompute) {
  for (Value operand : op->getOperands())
    cloneSlice(operand, rewriter, mapping, sourceBlock, op); // 克隆操作前先克隆它依赖的局部定义，保证新位置的 operand 可用。
  Operation *cloned = op->clone(mapping);
  rewriter.insert(cloned);
  mapping.map(op->getResults(), cloned->getResults()); // 建立原 SSA 值到克隆/替换值的映射，保证克隆操作使用新 buffer。
  if (markCompute) // compute 克隆需要打 db_plan_compute 标记，传输克隆另有 copy-role，二者不能混用。
    cloned->setAttr(kPlanComputeAttr, UnitAttr::get(op->getContext())); // 标记克隆出的计算，避免后续 pass 把它误当成原始传输。
  return cloned;
}

Operation *cloneTransfer(Operation *op, IRRewriter &rewriter, // 传输操作需要额外角色属性，供后续 lowering 区分 DMA start/store。
                         IRMapping &mapping, Block *sourceBlock,
                         StringRef role) {
  Operation *cloned = cloneMappedOp(op, rewriter, mapping, sourceBlock, // 按当前轮映射克隆 compute/store/load，保持原操作语义但替换 tile。
                                    /*markCompute=*/false);
  cloned->setAttr(kPlanCopyRoleAttr, rewriter.getStringAttr(role)); // 传输角色需要保留下来，后续 DMA lowering 才知道是 prefetch 还是 store。
  return cloned;
}

Value selectBuffer(IRRewriter &rewriter, Location loc, Value condition, // 用循环携带的布尔状态在 ping 和 pong 之间选择当前轮 buffer。
                   Value trueBuffer, Value falseBuffer) {
  return memref_ext::SelectMemrefOp::create(
             rewriter, loc, trueBuffer.getType(), condition, trueBuffer,
             falseBuffer)
      .getResult();
}

bool rewritePlan(IRRewriter &rewriter, Plan &plan, int64_t planId) { // 用 prologue 预取首轮、loop 内预取下一轮的模板替换原始 loop。
  scf::ForOp loop = plan.loop; // 保存要替换的原始循环，重写结束后会删除它。
  Location loc = loop.getLoc();
  Block *sourceBlock = loop.getBody();
  MLIRContext *context = loop.getContext();

  SmallVector<Value> pingBuffers;
  SmallVector<Value> pongBuffers;
  SmallVector<bool> needsDoubleBuffer;
  SmallVector<MemRefType> tileTypes;
  rewriter.setInsertionPoint(loop);
  auto sharedMemorySpace = // double-buffer 的本地 tile 放到 shared memory address space。
      rewriter.getI64IntegerAttr(triton::kSharedMemorySpace);
  for (TileInfo tile : plan.tiles) {
    auto tileType = dyn_cast<MemRefType>(tile.root.getType());
    if (!tileType || tileType.getRank() == 0 || !tileType.hasStaticShape()) // ping/pong buffer 由 memref.alloc 直接生成，必须知道静态形状；标量或动态 shape 无法安全套用模板。
      return false; // 当前 IR 形态无法证明安全，放弃这次优化以保持语义。
    auto bufferType = MemRefType::get(tileType.getShape(), // 保留原 tile 的 shape/layout/element type，只替换到 shared memory。
                                      tileType.getElementType(),
                                      tileType.getLayout(), sharedMemorySpace); // double-buffer 的本地 tile 放到 shared memory address space。
    pingBuffers.push_back( // 每个 tile 至少需要一份当前轮 buffer。
        memref::AllocOp::create(rewriter, loc, bufferType, ValueRange{}));
    if (tile.load) { // 有 load_ex 的输入 tile 需要两份 buffer，当前轮 compute 和下一轮 prefetch 才能并行而不互相覆盖。
      pongBuffers.push_back(
          memref::AllocOp::create(rewriter, loc, bufferType, ValueRange{}));
      needsDoubleBuffer.push_back(true); // 标记该 tile 在 steady-state 中需要 ping/pong 选择。
    } else {
      pongBuffers.push_back(Value{}); // store-only 或临时 tile 不需要下一轮预取，所以没有 pong buffer。
      needsDoubleBuffer.push_back(false); // 标记该 tile 始终使用同一份 buffer。
    }
    tileTypes.push_back(bufferType);
  }

  auto idAttr = IntegerAttr::get(IntegerType::get(context, 64), planId); // 计划 id 写入生成 IR，方便跨区域关联同一次重写。
  Value trueValue = arith::ConstantOp::create(rewriter, loc, // 循环初始选择 ping buffer，随后每轮翻转。
                                              rewriter.getBoolAttr(true));
  Value first = loop.getLowerBound();
  Value hasFirst = // 空循环不能执行 prologue 预取，避免越界 DMA。
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, first,
                            loop.getUpperBound());
  auto prologue = scf::IfOp::create(rewriter, loc, hasFirst, // 空循环不能执行 prologue 预取，避免越界 DMA。
                                    /*withElseRegion=*/false);
  prologue->setAttr(kPlanIdAttr, idAttr); // 给新生成的 prologue/loop 打同一个 id，方便后续 pass 识别同一计划。
  prologue->setAttr(kPlanPrologueAttr, UnitAttr::get(context)); // 标记首轮预取区域，后续 lower 能区分它和 steady-state 预取。
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(prologue.thenBlock()); // prologue 在进入 steady-state 前预取第 0 轮数据。
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), first); // 把原循环 iv 映射到首轮下标，克隆首轮 load 时复用原地址计算。
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      mapping.map(tile.root, pingBuffers[index]); // 首轮预取写入 ping buffer，供第一轮计算读取。
    for (auto [index, iterArg] : llvm::enumerate(loop.getRegionIterArgs()))
      mapping.map(iterArg, loop.getInitArgs()[index]); // prologue 在 loop 外部，iter_arg 必须映射到 init_arg 以避免引用被删除的 block argument。
    for (TileInfo tile : plan.tiles)
      if (tile.load) // prologue 只克隆输入 tile 的 load_ex，store-only tile 没有首轮预取来源。
        cloneTransfer(tile.load, rewriter, mapping, sourceBlock, // 传输操作需要额外角色属性，供后续 lowering 区分 DMA start/store。
                      kPrefetchRole);
  }

  SmallVector<Value> loopInits;
  loopInits.push_back(trueValue); // 第一个 iter_arg 保存 ping/pong 状态。
  for (Value init : loop.getInitArgs())
    loopInits.push_back(init); // 其余 iter_arg 透传原 loop 的循环携带初值。

  auto dbLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(), // 新循环携带 ping/pong 状态 + 原 iter_args，替代原始 loop 执行 steady-state。
                                   loop.getUpperBound(), loop.getStep(),
                                   loopInits); // 循环初始选择 ping buffer，随后每轮翻转。
  dbLoop->setAttr(kPlanIdAttr, idAttr); // 给新生成的 prologue/loop 打同一个 id，方便后续 pass 识别同一计划。
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(dbLoop.getBody());
    Value cur = dbLoop.getRegionIterArgs().front(); // 当前布尔状态决定本轮使用 ping 还是 pong。
    Value next = arith::XOrIOp::create(rewriter, loc, cur, trueValue); // 循环初始选择 ping buffer，随后每轮翻转。
    SmallVector<Value> currentViews; // 当前视图提供给本轮 wait 后的 compute/store。
    SmallVector<Value> nextViews; // 下一轮视图提供给提前发起的 prefetch。
    for (auto [index, pingBuffer] : llvm::enumerate(pingBuffers)) { // 为每个 tile 生成当前轮和下一轮视图，支撑 ping/pong 切换。
      if (!needsDoubleBuffer[index]) { // 没有输入预取的 tile 没必要分配第二份缓冲，复用同一个 buffer 即可。
        currentViews.push_back(pingBuffer); // 当前视图提供给本轮 wait 后的 compute/store。
        nextViews.push_back(pingBuffer); // 下一轮视图提供给提前发起的 prefetch。
        continue;
      }
      Value pongBuffer = pongBuffers[index];
      currentViews.push_back( // 当前视图提供给本轮 wait 后的 compute/store。
          selectBuffer(rewriter, loc, cur, pingBuffer, pongBuffer)); // 用循环携带的布尔状态在 ping 和 pong 之间选择当前轮 buffer。
      nextViews.push_back( // 下一轮视图提供给提前发起的 prefetch。
          selectBuffer(rewriter, loc, next, pingBuffer, pongBuffer)); // 用循环携带的布尔状态在 ping 和 pong 之间选择下一轮 buffer。
    }

    Value nextIv = // 下一轮下标用于克隆原 load 的地址和边界计算。
        arith::AddIOp::create(rewriter, loc, dbLoop.getInductionVar(),
                              dbLoop.getStep());
    Value hasNext = // 最后一轮没有下一块数据，必须跳过循环内预取。
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, nextIv, // 下一轮下标用于克隆原 load 的地址和边界计算。
                              dbLoop.getUpperBound());

    IRMapping currentMapping; // 当前轮映射把原 tile 根替换成本轮选中的 buffer。
    currentMapping.map(loop.getInductionVar(), dbLoop.getInductionVar()); // 当前轮映射把原 tile 根替换成本轮选中的 buffer。
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      currentMapping.map(tile.root, currentViews[index]); // 当前视图提供给本轮 wait 后的 compute/store。
    ValueRange origIterArgs = loop.getRegionIterArgs();
    ValueRange newIterArgs = dbLoop.getRegionIterArgs();
    for (auto [index, iterArg] : llvm::enumerate(origIterArgs))
      currentMapping.map(iterArg, newIterArgs[index + 1]); // +1 跳过 ping/pong 状态 iter_arg。

    // 把纯地址计算 op（如 block_desc_advance）提前克隆，让 prefetch 能用 advance 后的 descriptor 发起下一轮 DMA。
    // 这些 op 无 tile 内存副作用，提前执行不影响计算语义；后续 steadyOps 循环中跳过它们避免重复克隆。
    DenseSet<Operation *> clonedBeforePrefetch;
    for (Operation *op : plan.steadyOps) {
      if (getTransferLike(op)) // load/store 由专门路径处理，不属于地址计算。
        continue;
      if (!isMemoryEffectFree(op)) // 有内存副作用的 op 不能提前，否则会改变 tile 数据流的时序。
        continue;
      bool touchesTile = false;
      for (Value result : op->getResults()) {
        for (const TileInfo &tile : plan.tiles) {
          if (result == tile.root) {
            touchesTile = true;
            break;
          }
        }
        if (touchesTile)
          break;
      }
      if (touchesTile) // 结果是某个 tile 根内存时不能提前，避免和 ping/pong 选择冲突。
        continue;
      cloneMappedOp(op, rewriter, currentMapping, sourceBlock, /*markCompute=*/true);
      clonedBeforePrefetch.insert(op);
    }

    auto prefetch = scf::IfOp::create(rewriter, loc, hasNext, // 最后一轮没有下一块数据，必须跳过循环内预取。
                                      /*withElseRegion=*/false);
    prefetch->setAttr(kPlanPrefetchAttr, UnitAttr::get(context)); // 标记循环内下一轮预取区域，帮助 DMA lowering 做角色区分。
    {
      OpBuilder::InsertionGuard ifGuard(rewriter);
      rewriter.setInsertionPointToStart(prefetch.thenBlock()); // prefetch 在 compute/store 之前发起 DMA，实现与当前轮计算的重叠。
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), nextIv); // 下一轮下标用于克隆原 load 的地址和边界计算。
      for (auto [index, tile] : llvm::enumerate(plan.tiles))
        mapping.map(tile.root, nextViews[index]); // 下一轮视图提供给提前发起的 prefetch。
      auto origYieldForPrefetch = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      for (auto [index, iterArg] : llvm::enumerate(loop.getRegionIterArgs()))
        mapping.map(iterArg, currentMapping.lookupOrDefault(origYieldForPrefetch.getOperand(index))); // yield 的第 N 个操作数就是下一轮的第 N 个 iter_arg，通过 currentMapping 查到 advance 后的克隆结果。
      for (TileInfo tile : plan.tiles)
        if (tile.load) // steady-state 只为有 load_ex 的 tile 预取下一轮，避免为 store-only tile 生成无意义 DMA。
          cloneTransfer(tile.load, rewriter, mapping, sourceBlock, // 传输操作需要额外角色属性，供后续 lowering 区分 DMA start/store。
                        kPrefetchRole);
    }

    for (Operation *op : plan.steadyOps) { // 按原顺序克隆 compute/store/scratch，保持副作用次序与数据流语义。
      if (clonedBeforePrefetch.count(op)) // 已在 prefetch 前克隆的地址计算 op 不重复克隆。
        continue;
      std::optional<TransferLike> transfer = getTransferLike(op);
      if (transfer && transfer->isStore()) // 写回 store 需要 copy-role，供后续 lowering 区分。
        cloneTransfer(op, rewriter, currentMapping, sourceBlock, kDB2StoreRole);
      else
        cloneMappedOp(op, rewriter, currentMapping, sourceBlock, // 按当前轮映射克隆，替换 tile 与 iter_arg。
                      /*markCompute=*/true);
    }

    SmallVector<Value> yieldValues;
    yieldValues.push_back(next); // 翻转后的 ping/pong 状态传给下一轮。
    auto origYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (Value operand : origYield.getOperands()) // 原循环携带值经当前轮映射后继续 yield。
      yieldValues.push_back(currentMapping.lookupOrDefault(operand));
    scf::YieldOp::create(rewriter, loc, yieldValues); // 完成 ping/pong 轮换并透传 iter_args。
  }
  for (auto [index, result] : llvm::enumerate(loop.getResults())) // 原 loop 结果全部改由新 loop 提供（跳过状态结果）。
    result.replaceAllUsesWith(dbLoop.getResult(index + 1));
  rewriter.eraseOp(loop); // 新结构已完整替代原 loop，删除旧 loop 避免重复执行。
  return true;
}

struct HexagonDoubleBufferPlanRewriteExtPass
    : public ::impl::HexagonDoubleBufferPlanRewriteExtBase<
          HexagonDoubleBufferPlanRewriteExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override { // 重写会生成 arith/scf/memref/memref_ext/triton 操作，必须注册方言。
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    memref_ext::MemRefExtDialect, scf::SCFDialect,
                    triton::TritonDialect>();
  }

  void runOnOperation() override {
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>(); // 依赖构建用别名分析决定哪些操作可能访问同一 tile。
    int64_t nextPlanId = 0; // 每个被重写 loop 拿到唯一计划 id，便于后续识别。
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) { loops.push_back(loop); }); // 先收集 loop 再重写，避免 walk 过程中删除/替换 IR 破坏遍历。
    for (scf::ForOp loop : loops) {
      auto kind = loop->getAttrOfType<StringAttr>(kLoopKindAttr);
      if (!kind || (kind.getValue() != kPointwise && kind.getValue() != kReduce)) // 只有 pointwise/reduce loop 匹配当前双缓冲模板，未标记或其他 kind 的循环保持原样。
        continue;
      Plan plan;
      plan.isReduction = kind.getValue() == kReduce; // reduce loop 可以没有显式 store，分类条件需要放宽。
      ExtTileDataFlowAnalysis analysis(loop, aliasAnalysis);
      if (!analysis.run(plan)) // 分析失败表示 tile 根、依赖顺序或 load/compute/store 分类不完整，保守保留原 loop。
        continue;
      IRRewriter rewriter(loop.getContext());
      rewritePlan(rewriter, plan, nextPlanId++); // 用 prologue 预取首轮、loop 内预取下一轮的模板替换原始 loop。
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferPlanRewriteExtPass() {
  return std::make_unique<HexagonDoubleBufferPlanRewriteExtPass>(); // 把 pass 实例交给 MLIR pass pipeline 管理生命周期。
}

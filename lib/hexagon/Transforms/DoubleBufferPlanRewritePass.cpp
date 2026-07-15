//===- DoubleBufferPlanRewritePass.cpp -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Transforms/CopyDirection.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERPLANREWRITE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// 以下属性用于在 plan rewrite 和后续 DMA lowering pass 之间传递内部计划。
constexpr StringLiteral kPlanIdAttr = "db_plan_id";
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue";
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch";
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute";
constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role";

// tile 访问类型使用 bit 位表示，便于组合读写。
enum TileAccessKind : uint8_t {
  // 当前操作不访问该 tile。
  NoTileAccess = 0,
  // 当前操作读取该 tile。
  TileRead = 1,
  // 当前操作写入该 tile。
  TileWrite = 2,
};

// 数据流图里的一个操作节点。
struct TileNode {
  // 原始 MLIR operation。
  Operation *op = nullptr;
  // 每个 tile 对应一个 bitmask，记录该操作对 tile 的读写。
  SmallVector<uint8_t> accesses;
  // 当前节点依赖的前驱节点索引。
  SmallVector<unsigned> predecessors;
  // 是否是 global -> shared 的 load copy。
  bool isLoad = false;
  // 是否是 load 之前用于初始化/覆盖 tile 的 setup 操作。
  bool isLoadSetup = false;
  // 是否是主要计算操作。
  bool isCompute = false;
  // 是否是 shared -> global 的 store copy。
  bool isStore = false;
};

// 描述一个可双缓冲的 tile。
struct TileInfo {
  // tile 的根 memref。
  Value root;
  // 给该 tile 从 global 预取数据的 copy。
  memref::CopyOp load;
  // 每次 load 前需要一起克隆的初始化/setup 操作。
  SmallVector<Operation *> setupOps;
};

// 一个循环的 double-buffer 重写计划。
struct Plan {
  // 被重写的原始 scf.for。
  scf::ForOp loop;
  // 参与 double-buffer 的 tile 列表。
  SmallVector<TileInfo> tiles;
  // 原循环体的数据流节点。
  SmallVector<TileNode> nodes;
  // 需要放到当前 tile 上执行的计算操作。
  SmallVector<Operation *> computes;
  // 需要在计算后写回 global 的 store copy。
  SmallVector<memref::CopyOp> stores;
  // 该循环是否被前置 pass 标记为 reduction。
  bool isReduction = false;
};

// 判断 copy 是否带有指定 direction。
bool hasCopyDirection(memref::CopyOp copy, StringRef direction) {
  // direction 属性由 copy direction pass 写入。
  auto attr = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  // 属性存在且字符串匹配才返回 true。
  return attr && attr.getValue() == direction;
}

// 查找 value 对应的 tile 根，只接受 alloc/view 等可作为本地 tile 的根。
Value findMemoryRoot(Value value) {
  // 循环剥离 subview/cast/reinterpret_cast。
  while (true) {
    // 空 value 没有根。
    if (!value)
      return {};
    // alloc 的结果就是 tile 根。
    if (auto alloc = value.getDefiningOp<memref::AllocOp>())
      return alloc.getMemref();
    // view 的结果在这里被视为可独立匹配的根。
    if (auto view = value.getDefiningOp<memref::ViewOp>())
      return view.getResult();
    // 没有 defining op 的值不是本 pass 可重写的本地 tile。
    Operation *def = value.getDefiningOp();
    if (!def)
      return {};
    // subview 继续追踪 source。
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    // cast 不改变底层存储，继续追踪 source。
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    // reinterpret_cast 同样继续追踪 source。
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      continue;
    }
    // 其它定义不是可识别 tile 根。
    return {};
  }
}

// 查找用于 global load/store 独立性判断的别名根。
Value findAliasRoot(Value value) {
  // 沿着 view-like 操作规约 value。
  while (value) {
    // block argument 或外部值直接作为别名根。
    Operation *def = value.getDefiningOp();
    if (!def)
      return value;
    // subview 的别名根来自 source。
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    // cast 的别名根来自 source。
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    // reinterpret_cast 的别名根来自 source。
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      continue;
    }
    // view 在当前层级作为别名根返回。
    if (auto view = dyn_cast<memref::ViewOp>(def))
      return view.getResult();
    // 其它定义无法继续剥离，当前 value 作为根。
    return value;
  }
  // 输入为空时返回空 value。
  return {};
}

// 判断 tile 的定义在 copy 执行前是否可用。
bool isAvailableBeforeCopy(Value tile, scf::ForOp loop, memref::CopyOp copy) {
  // 必须能找到 tile 的定义操作。
  Operation *def = tile.getDefiningOp();
  if (!def)
    return false;
  // 如果 tile 定义在循环体内，它必须位于 load copy 之前。
  Block *body = loop.getBody();
  if (def->getBlock() == body)
    return def->isBeforeInBlock(copy);
  // 如果 tile 定义在循环外，它必须位于 loop 之前。
  return def->getBlock() == loop->getBlock() && def->isBeforeInBlock(loop);
}

// 将 tile access bitmask 转成调试输出字符串。
StringRef accessName(uint8_t access) {
  if (access == (TileRead | TileWrite))
    return "READ_WRITE";
  if (access == TileRead)
    return "READ";
  if (access == TileWrite)
    return "WRITE";
  return "NONE";
}

// 针对一个 scf.for 构建 tile 数据流并生成 Plan。
struct TileDataFlowAnalysis {
  // 保存目标 loop 和别名分析。
  TileDataFlowAnalysis(scf::ForOp loop, AliasAnalysis &aliasAnalysis)
      : loop(loop), aliasAnalysis(aliasAnalysis) {}

  // 运行完整分析流程。
  bool run(Plan &plan) {
    // 记录目标循环。
    plan.loop = loop;
    // 依次收集 tile、构建节点、分类节点、验证全局 load/store 不冲突。
    if (!collectTiles(plan) || !buildNodes(plan) || !classify(plan) ||
        !validateGlobalLoadStoreIndependence(plan))
      return false;
    // 打印分析结果，便于调试重写计划。
    print(plan, llvm::outs());
    return true;
  }

  // 查询某个 value 命中的 tile 索引。
  std::optional<unsigned> getTileIndex(Value value, const Plan &plan) {
    // 先规约到 tile 根。
    value = findMemoryRoot(value);
    if (!value)
      return std::nullopt;
    // 用相等或 aliasAnalysis 判断是否属于某个 tile。
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      if (value == tile.root || aliasAnalysis.alias(value, tile.root))
        return index;
    // 没有命中任何 tile。
    return std::nullopt;
  }

  // 从循环体里的 global->shared copy 收集可双缓冲 tile。
  bool collectTiles(Plan &plan) {
    // 扫描循环体内非 terminator 操作。
    for (Operation &op : loop.getBody()->without_terminator()) {
      // 只关心 global->shared 的 memref.copy。
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy || !hasCopyDirection(copy, kGlobalToShared))
        continue;

      // load copy 的 target 应该对应本地 tile。
      Value root = findMemoryRoot(copy.getTarget());
      // tile 必须存在，并且在 copy 前已经定义。
      if (!root || !isAvailableBeforeCopy(root, loop, copy))
        return false;
      // 一个 tile 只能有一个 load copy，否则计划不唯一。
      if (llvm::any_of(plan.tiles,
                       [&](TileInfo tile) { return tile.root == root; }))
        return false;
      // 记录 tile 根和对应 load copy。
      plan.tiles.push_back(TileInfo{root, copy});
    }
    // 至少需要一个 tile 才有 double-buffer 计划。
    return !plan.tiles.empty();
  }

  // 根据 memref.copy 的 source/target 给节点补充 tile 读写信息。
  void addCopyAccesses(memref::CopyOp copy, TileNode &node, Plan &plan) {
    // source 命中 tile 表示读 tile。
    if (auto source = getTileIndex(copy.getSource(), plan))
      node.accesses[*source] |= TileRead;
    // target 命中 tile 表示写 tile。
    if (auto target = getTileIndex(copy.getTarget(), plan))
      node.accesses[*target] |= TileWrite;
  }

  // 从 MemoryEffectOpInterface 精确提取操作对 tile 的读写。
  bool addEffectAccesses(Operation *op, TileNode &node, Plan &plan) {
    // 没有 effect interface 时无法精确分析。
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return false;

    // 读取所有 memory effects。
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      // 只关心读写 effect。
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
      if (!isRead && !isWrite)
        continue;

      // effect 必须绑定到 memref value，否则无法精确匹配 tile。
      Value value = effect.getValue();
      if (!value || !isa<BaseMemRefType>(value.getType()))
        return false;

      // 命中 tile 后记录读写 bit。
      if (auto tile = getTileIndex(value, plan)) {
        if (isRead)
          node.accesses[*tile] |= TileRead;
        if (isWrite)
          node.accesses[*tile] |= TileWrite;
      }
    }
    return true;
  }

  // effect 精确分析失败时，用 alias analysis 做保守读写推断。
  void addConservativeAccesses(Operation *op, TileNode &node, Plan &plan) {
    // 对每个 tile 分别查询 op 是否可能 mod/ref。
    for (auto [index, tile] : llvm::enumerate(plan.tiles)) {
      ModRefResult modRef = aliasAnalysis.getModRef(op, tile.root);
      // Ref 表示可能读取该 tile。
      if (modRef.isRef())
        node.accesses[index] |= TileRead;
      // Mod 表示可能写入该 tile。
      if (modRef.isMod())
        node.accesses[index] |= TileWrite;
    }
  }

  // 构建循环体内与 tile 相关的节点和依赖。
  bool buildNodes(Plan &plan) {
    // 每个 tile 最近一次写节点。
    SmallVector<std::optional<unsigned>> lastWrite(plan.tiles.size());
    // 每个 tile 最近一次写之后出现的读节点。
    SmallVector<SmallVector<unsigned>> readsSinceWrite(plan.tiles.size());

    // 顺序扫描循环体操作。
    for (Operation &op : loop.getBody()->without_terminator()) {
      // 初始化当前节点的访问 bitmask。
      TileNode node;
      node.op = &op;
      node.accesses.assign(plan.tiles.size(), NoTileAccess);

      // copy 用 source/target 判定访问。
      if (auto copy = dyn_cast<memref::CopyOp>(&op))
        addCopyAccesses(copy, node, plan);
      // 常见 memref 视图/alloc/dealloc 或无副作用操作通常不直接访问数据。
      else if (!isa<memref::AllocOp, memref::DeallocOp, memref::SubViewOp,
                    memref::ViewOp, memref::CastOp,
                    memref::ReinterpretCastOp>(&op) &&
               !isMemoryEffectFree(&op)) {
        // 对有副作用的普通操作优先精确分析，失败后保守分析。
        if (!addEffectAccesses(&op, node, plan))
          addConservativeAccesses(&op, node, plan);
      } else if (!isMemoryEffectFree(&op) &&
                 !addEffectAccesses(&op, node, plan)) {
        // 理论上少数有 effect 的视图类操作也保守处理。
        addConservativeAccesses(&op, node, plan);
      }

      // 完全不访问 tile 的操作不进入计划。
      if (!llvm::any_of(node.accesses,
                        [](uint8_t access) { return access != NoTileAccess; }))
        continue;

      // copy 节点额外标记 load/store 角色。
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        node.isLoad = hasCopyDirection(copy, kGlobalToShared);
        node.isStore = hasCopyDirection(copy, kSharedToGlobal);
      }

      // 根据读写顺序构建当前节点的前驱集合。
      unsigned nodeIndex = plan.nodes.size();
      llvm::SmallDenseSet<unsigned> predecessors;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        // 拆出读写 bit。
        bool reads = access & TileRead;
        bool writes = access & TileWrite;
        // 读依赖最近一次写。
        if (reads && lastWrite[tileIndex])
          predecessors.insert(*lastWrite[tileIndex]);
        // 写依赖最近一次写和最近一次写之后的所有读。
        if (writes) {
          if (lastWrite[tileIndex])
            predecessors.insert(*lastWrite[tileIndex]);
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          // 当前写刷新该 tile 的读写状态。
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
          if (reads)
            readsSinceWrite[tileIndex].push_back(nodeIndex);
        } else if (reads) {
          // 纯读记录下来，供后续写建立 WAR 依赖。
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      // 将 set 转为排序 vector，保证计划稳定。
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors);
      // 保存当前节点。
      plan.nodes.push_back(std::move(node));
    }
    return true;
  }

  // 判断一个操作是否属于可在 load 前重复执行的覆盖式 setup。
  bool isOverwriteFillSetupOp(Operation *op) {
    // linalg.fill 明确是覆盖式写入。
    if (isa<linalg::FillOp>(op))
      return true;

    // 也允许只包含 fill-like 操作的无 else scf.if。
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || !ifOp.getElseRegion().empty())
      return false;

    // 遍历 if 内部确认存在 fill，且其它操作都无副作用或是 yield。
    bool foundFill = false;
    bool onlyFillLikeOps = true;
    ifOp.walk([&](Operation *nested) {
      if (nested == op || isa<scf::YieldOp>(nested))
        return WalkResult::advance();
      if (isa<linalg::FillOp>(nested)) {
        foundFill = true;
        return WalkResult::advance();
      }
      if (isMemoryEffectFree(nested))
        return WalkResult::advance();
      onlyFillLikeOps = false;
      return WalkResult::interrupt();
    });
    // 必须找到 fill，且没有其它有副作用操作。
    return foundFill && onlyFillLikeOps;
  }

  // 如果节点只覆盖写入单个 tile，则返回该 tile 索引。
  std::optional<unsigned> getSingleSetupTile(const TileNode &node) {
    // setupTile 用来记录唯一候选 tile。
    std::optional<unsigned> setupTile;
    for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
      if (access == NoTileAccess)
        continue;
      // 纯写，或者 fill-like 的读写组合，都可视为覆盖式 setup。
      bool overwritesTile = access == TileWrite ||
                            (access == (TileRead | TileWrite) &&
                             isOverwriteFillSetupOp(node.op));
      if (!overwritesTile)
        return std::nullopt;
      // setup 操作只能对应一个 tile。
      if (setupTile)
        return std::nullopt;
      setupTile = tileIndex;
    }
    // 返回唯一 tile，或空。
    return setupTile;
  }

  // 判断某个节点之后是否还有 load 会写同一个 tile。
  bool hasFollowingLoadForTile(unsigned nodeIndex, unsigned tileIndex,
                               const Plan &plan) {
    // 只检查当前节点后面的节点。
    for (unsigned nextIndex = nodeIndex + 1; nextIndex < plan.nodes.size();
         ++nextIndex) {
      const TileNode &next = plan.nodes[nextIndex];
      // 只关心 load 节点。
      if (!next.isLoad)
        continue;
      // load 写该 tile，说明当前 setup 可附着到该 load 前。
      if (next.accesses[tileIndex] & TileWrite)
        return true;
    }
    // 后续没有对应 load。
    return false;
  }

  // 将数据流节点分类为 load/setup/store/compute，并填充 plan。
  bool classify(Plan &plan) {
    // 按原始顺序处理节点。
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes)) {
      // load 节点必须真的是 memref.copy。
      if (node.isLoad) {
        if (!isa<memref::CopyOp>(node.op))
          return false;
        continue;
      }
      // store 节点需要从 tile 写向非 tile 目标。
      if (node.isStore) {
        auto copy = cast<memref::CopyOp>(node.op);
        if (!getTileIndex(copy.getSource(), plan) ||
            getTileIndex(copy.getTarget(), plan))
          return false;
        plan.stores.push_back(copy);
        continue;
      }
      // load 前的覆盖式 setup 会被记录到对应 tile。
      if (auto setupTile = getSingleSetupTile(node);
          setupTile && hasFollowingLoadForTile(nodeIndex, *setupTile, plan)) {
        node.isLoadSetup = true;
        plan.tiles[*setupTile].setupOps.push_back(node.op);
        continue;
      }
      // 其它 memref.copy 不属于可支持的计划。
      if (isa<memref::CopyOp>(node.op))
        return false;
      // 剩余 tile 相关操作作为 compute 克隆到新循环体。
      node.isCompute = true;
      plan.computes.push_back(node.op);
    }
    // pointwise 至少需要 compute 和 store；reduction 允许没有 store。
    return !plan.computes.empty() && (plan.isReduction || !plan.stores.empty());
  }

  // 验证 global load source 和 store target 不 must-alias，避免覆盖输入。
  bool validateGlobalLoadStoreIndependence(const Plan &plan) {
    // 对每个 tile 的 load source 检查所有 store target。
    for (TileInfo tile : plan.tiles) {
      Value loadSource = findAliasRoot(tile.load.getSource());
      if (!loadSource)
        return false;
      for (memref::CopyOp store : plan.stores) {
        Value storeTarget = findAliasRoot(store.getTarget());
        // 源和目标相同或 must-alias 时，double-buffer 改写可能改变语义。
        if (!storeTarget || loadSource == storeTarget ||
            aliasAnalysis.alias(loadSource, storeTarget).isMust())
          return false;
      }
    }
    // 没有发现必须别名冲突。
    return true;
  }

  // 打印 plan 的节点和数据依赖。
  void print(const Plan &plan, raw_ostream &os) {
    // 打印数据流分析标题。
    os << "\n=== HexagonDoubleBufferPlanRewrite dataflow ===\n";
    // 打印当前 loop 的概要，不展开 region。
    os << "loop: ";
    loop->print(os, OpPrintingFlags().skipRegions());
    os << "\n";
    // 打印每个节点的分类和 tile 访问。
    os << "nodes:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes)) {
      os << "  [" << nodeIndex << "] ";
      if (node.isLoad)
        os << "LOAD";
      else if (node.isLoadSetup)
        os << "LOAD_SETUP";
      else if (node.isStore)
        os << "STORE";
      else
        os << "COMPUTE";
      os << " accesses={";
      bool first = true;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        if (access == NoTileAccess)
          continue;
        if (!first)
          os << ", ";
        first = false;
        os << "tile#" << tileIndex << ":" << accessName(access);
      }
      os << "}\n      op: ";
      node.op->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }

    // 打印数据依赖边。
    os << "dataflow:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes)) {
      for (unsigned predecessor : node.predecessors) {
        const TileNode &source = plan.nodes[predecessor];
        for (auto [tileIndex, sourceAccess] : llvm::enumerate(source.accesses)) {
          // 通过前驱访问和当前访问重新计算依赖类型，便于输出 RAW/WAR/WAW。
          uint8_t targetAccess = node.accesses[tileIndex];
          bool raw = (sourceAccess & TileWrite) && (targetAccess & TileRead);
          bool war = (sourceAccess & TileRead) && (targetAccess & TileWrite);
          bool waw = (sourceAccess & TileWrite) && (targetAccess & TileWrite);
          if (!raw && !war && !waw)
            continue;
          os << "  [" << predecessor << "] -> [" << nodeIndex << "] ";
          if (raw)
            os << "RAW";
          if (war)
            os << (raw ? "+WAR" : "WAR");
          if (waw)
            os << ((raw || war) ? "+WAW" : "WAW");
          os << "(tile#" << tileIndex << ")\n";
        }
      }
    }
    os << "=== End HexagonDoubleBufferPlanRewrite dataflow ===\n";
  }

  // 被分析的循环。
  scf::ForOp loop;
  // 用于判断 view/alias/modref 的别名分析。
  AliasAnalysis &aliasAnalysis;
};

// 判断 op 是否定义在指定 block 中且位于 limit 之前。
bool isDefinedBefore(Operation *op, Block *block, Operation *limit) {
  return op && op->getBlock() == block && op->isBeforeInBlock(limit);
}

// 克隆产生某个 value 所需的地址计算切片。
Value cloneSlice(Value value, IRRewriter &rewriter, IRMapping &mapping,
                 Block *sourceBlock, Operation *limit) {
  // 如果 value 已经被映射，直接复用映射后的 value。
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  // 只克隆同一 sourceBlock 中、位于 limit 之前的定义操作。
  Operation *def = value.getDefiningOp();
  if (!isDefinedBefore(def, sourceBlock, limit))
    return value;

  // 先递归克隆定义操作依赖的操作数。
  for (Value operand : def->getOperands())
    cloneSlice(operand, rewriter, mapping, sourceBlock, limit);

  // 基于当前 mapping 克隆定义操作。
  Operation *cloned = def->clone(mapping);
  // subview 克隆后重新推断结果类型，避免源 memref 类型变化后结果类型过期。
  if (auto subview = dyn_cast<memref::SubViewOp>(cloned)) {
    auto sourceType = cast<MemRefType>(subview.getSource().getType());
    subview.getResult().setType(memref::SubViewOp::inferResultType(
        sourceType, subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides()));
  }
  // 将克隆操作插入到当前 rewriter 插入点。
  rewriter.insert(cloned);
  // 建立原结果到克隆结果的映射。
  mapping.map(def->getResults(), cloned->getResults());
  // 返回请求 value 的克隆结果。
  return mapping.lookup(value);
}

// 克隆一个 memref.copy，并额外打上 plan copy role。
memref::CopyOp cloneCopy(memref::CopyOp copy, IRRewriter &rewriter,
                         IRMapping &mapping, Block *sourceBlock,
                         StringRef role) {
  // 克隆 source 地址计算切片。
  Value source = cloneSlice(copy.getSource(), rewriter, mapping, sourceBlock,
                            copy.getOperation());
  // 克隆 target 地址计算切片。
  Value target = cloneSlice(copy.getTarget(), rewriter, mapping, sourceBlock,
                            copy.getOperation());
  // 用映射后的 source/target 创建新的 memref.copy。
  auto cloned = memref::CopyOp::create(rewriter, copy.getLoc(), source, target);
  // 保留原 copy direction 属性。
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    cloned->setAttr(kCopyDirectionAttrName, direction);
  // 标记 copy 在 double-buffer 计划中的角色。
  cloned->setAttr(kPlanCopyRoleAttr, rewriter.getStringAttr(role));
  // 返回新 copy。
  return cloned;
}

// 克隆普通操作，并根据需要标记为 compute。
Operation *cloneMappedOp(Operation *op, IRRewriter &rewriter,
                         IRMapping &mapping, Block *sourceBlock,
                         bool markCompute) {
  // 先确保所有操作数需要的定义切片都已经克隆或映射。
  for (Value operand : op->getOperands())
    cloneSlice(operand, rewriter, mapping, sourceBlock, op);
  // 基于现有 mapping 克隆操作本体。
  Operation *cloned = op->clone(mapping);
  // 插入克隆操作。
  rewriter.insert(cloned);
  // 记录结果映射，供后续克隆操作使用。
  mapping.map(op->getResults(), cloned->getResults());
  // compute 操作打内部属性，后续 DMA lowering 清理。
  if (markCompute)
    cloned->setAttr(kPlanComputeAttr, UnitAttr::get(op->getContext()));
  // 返回克隆操作。
  return cloned;
}

// 从 double-sized backing buffer 中创建单个 ping/pong slot 的 subview。
Value createSlotView(IRRewriter &rewriter, Location loc, Value backing,
                     MemRefType tileType, Value offset) {
  // 每个维度默认从 0 开始。
  SmallVector<OpFoldResult> offsets(tileType.getRank(), rewriter.getIndexAttr(0));
  // sizes 使用原 tile 的完整 shape。
  SmallVector<OpFoldResult> sizes;
  // stride 默认全 1。
  SmallVector<OpFoldResult> strides(tileType.getRank(), rewriter.getIndexAttr(1));
  // 第一维 offset 选择 ping 或 pong 段。
  offsets.front() = offset;
  // 将静态 shape 转为 subview sizes。
  for (int64_t dim : tileType.getShape())
    sizes.push_back(rewriter.getIndexAttr(dim));
  // 从 backing 类型推断 subview 结果类型。
  auto backingType = cast<MemRefType>(backing.getType());
  auto viewType = memref::SubViewOp::inferResultType(backingType, offsets,
                                                     sizes, strides);
  // 创建并返回 subview。
  return memref::SubViewOp::create(rewriter, loc, viewType, backing, offsets,
                                   sizes, strides);
}

// 将 pointwise/reduction 计划重写为显式 ping/pong double-buffer 循环。
bool rewritePointwisePlan(IRRewriter &rewriter, Plan &plan, int64_t planId) {
  // 取出原始循环和常用上下文。
  scf::ForOp loop = plan.loop;
  Location loc = loop.getLoc();
  Block *sourceBlock = loop.getBody();
  MLIRContext *context = loop.getContext();

  // backings 保存每个 tile 对应的双倍大小 buffer。
  SmallVector<Value> backings;
  // tileTypes 保存原始 tile 类型，后续创建 slot view 时复用。
  SmallVector<MemRefType> tileTypes;
  // backing alloc 插入到原 loop 之前。
  rewriter.setInsertionPoint(loop);
  // 维持原代码约定的 2048 字节对齐。
  auto alignmentAttr = rewriter.getI64IntegerAttr(2048);
  // 为每个 tile 创建一个第一维扩大 2 倍的 backing buffer。
  for (TileInfo tile : plan.tiles) {
    // 当前实现只支持静态 shape、rank > 0 的 memref tile。
    auto tileType = dyn_cast<MemRefType>(tile.root.getType());
    if (!tileType || tileType.getRank() == 0 || !tileType.hasStaticShape())
      return false;
    // 复制原 shape，并把第一维乘 2 来容纳 ping/pong 两个 slot。
    SmallVector<int64_t> shape(tileType.getShape().begin(),
                               tileType.getShape().end());
    shape.front() *= 2;
    // backing 保持 element/layout/memory space 与原 tile 一致。
    auto backingType =
        MemRefType::get(shape, tileType.getElementType(), tileType.getLayout(),
                        tileType.getMemorySpace());
    // 创建 backing alloc，并记录。
    backings.push_back(memref::AllocOp::create(
        rewriter, loc, backingType, ValueRange{}, alignmentAttr));
    tileTypes.push_back(tileType);
  }

  // 当前 plan id 用于把 prologue/kernel/prefetch 关联起来。
  auto idAttr = IntegerAttr::get(IntegerType::get(context, 64), planId);
  // bool true 用作 ping/pong 状态初始值和异或翻转常量。
  Value trueValue = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getBoolAttr(true));
  // 常量 0 用于首轮 ping slot 偏移。
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  // first 是原循环的第一轮 iv。
  Value first = loop.getLowerBound();

  // 原循环非空时才需要执行 prologue 预取。
  Value hasFirst =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, first,
                            loop.getUpperBound());
  // prologue if 负责预取第一个 tile。
  auto prologue = scf::IfOp::create(rewriter, loc, hasFirst,
                                    /*withElseRegion=*/false);
  // 给 prologue 打上 plan id 和角色属性。
  prologue->setAttr(kPlanIdAttr, idAttr);
  prologue->setAttr(kPlanPrologueAttr, UnitAttr::get(context));
  {
    // 进入 prologue then block 构造首轮预取。
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(prologue.thenBlock());
    // mapping 将原循环 iv 和 tile 映射到 prologue 中的新值。
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), first);
    // 首轮预取使用 offset 0 的 slot。
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      mapping.map(tile.root,
                  createSlotView(rewriter, loc, backings[index],
                                 tileTypes[index], zero));
    // 克隆每个 tile 的 setup 和 load copy。
    for (TileInfo tile : plan.tiles) {
      for (Operation *setup : tile.setupOps)
        cloneMappedOp(setup, rewriter, mapping, sourceBlock,
                      /*markCompute=*/false);
      cloneCopy(tile.load, rewriter, mapping, sourceBlock, kPrefetchRole);
    }
  }

  // dbLoop 的 iter_arg 保存当前 ping/pong 状态。
  SmallVector<Value> initArgs{trueValue};
  // 创建新的 double-buffer 主循环，边界和 step 复用原循环。
  auto dbLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(),
                                   loop.getUpperBound(), loop.getStep(),
                                   initArgs);
  // 给新循环打 plan id，供 DMA lowering 查找。
  dbLoop->setAttr(kPlanIdAttr, idAttr);

  {
    // 构造新循环体。
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(dbLoop.getBody());
    // cur 表示当前 ping/pong 状态。
    Value cur = dbLoop.getRegionIterArgs().front();
    // next 是下一轮状态，使用 xor true 翻转。
    Value next = arith::XOrIOp::create(rewriter, loc, cur, trueValue);
    // nextIndex 选择下一轮预取写入的 slot。
    Value nextIndex = arith::IndexCastUIOp::create(rewriter, loc,
                                                   rewriter.getIndexType(), cur);
    // curIndex 选择当前计算读取的 slot。
    Value curIndex = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getIndexType(), next);

    // 当前 slot view 和下一轮 slot view 分别保存。
    SmallVector<Value> currentViews;
    SmallVector<Value> nextViews;
    // 为每个 tile 计算当前/下一轮 slot 的 subview。
    for (auto [index, tileType] : llvm::enumerate(tileTypes)) {
      // 一个 slot 在 backing 第一维上的长度等于原 tile 第一维。
      Value tileSize = arith::ConstantIndexOp::create(
          rewriter, loc, tileType.getShape().front());
      // 当前 slot offset = curIndex * tileSize。
      Value currentOffset =
          arith::MulIOp::create(rewriter, loc, curIndex, tileSize);
      // 下一轮 slot offset = nextIndex * tileSize。
      Value nextOffset =
          arith::MulIOp::create(rewriter, loc, nextIndex, tileSize);
      // 创建当前计算使用的 view。
      currentViews.push_back(
          createSlotView(rewriter, loc, backings[index], tileType,
                         currentOffset));
      // 创建下一轮预取使用的 view。
      nextViews.push_back(
          createSlotView(rewriter, loc, backings[index], tileType, nextOffset));
    }

    // 计算下一轮原始 iv。
    Value nextIv =
        arith::AddIOp::create(rewriter, loc, dbLoop.getInductionVar(),
                              dbLoop.getStep());
    // 判断是否存在下一轮，存在才预取。
    Value hasNext =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, nextIv,
                              dbLoop.getUpperBound());
    // prefetch if 负责在当前轮计算前发起下一轮预取。
    auto prefetch = scf::IfOp::create(rewriter, loc, hasNext,
                                      /*withElseRegion=*/false);
    // 标记为 prefetch，供 DMA lowering 定位。
    prefetch->setAttr(kPlanPrefetchAttr, UnitAttr::get(context));
    {
      // 构造 prefetch then block。
      OpBuilder::InsertionGuard ifGuard(rewriter);
      rewriter.setInsertionPointToStart(prefetch.thenBlock());
      // mapping 将原 loop iv 映射为 nextIv，将 tile 映射为 nextViews。
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), nextIv);
      for (auto [index, tile] : llvm::enumerate(plan.tiles))
        mapping.map(tile.root, nextViews[index]);
      // 克隆每个 tile 的 setup 和 load copy，完成下一轮预取。
      for (TileInfo tile : plan.tiles) {
        for (Operation *setup : tile.setupOps)
          cloneMappedOp(setup, rewriter, mapping, sourceBlock,
                        /*markCompute=*/false);
        cloneCopy(tile.load, rewriter, mapping, sourceBlock, kPrefetchRole);
      }
    }

    // currentMapping 用于克隆当前轮 compute/store。
    IRMapping currentMapping;
    // 原 loop iv 映射到新循环当前 iv。
    currentMapping.map(loop.getInductionVar(), dbLoop.getInductionVar());
    // 原 tile 映射到当前 slot view。
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      currentMapping.map(tile.root, currentViews[index]);
    // 克隆所有 compute 操作，并打 compute 属性。
    for (Operation *compute : plan.computes)
      cloneMappedOp(compute, rewriter, currentMapping, sourceBlock,
                    /*markCompute=*/true);
    // 克隆所有 store copy，并标记为 DB2 store。
    for (memref::CopyOp store : plan.stores)
      cloneCopy(store, rewriter, currentMapping, sourceBlock, kDB2StoreRole);

    // yield 下一轮 ping/pong 状态。
    scf::YieldOp::create(rewriter, loc, next);
  }

  // 新结构构造完成后删除原循环。
  rewriter.eraseOp(loop);
  // 重写成功。
  return true;
}

// Pass 主体：识别可 double-buffer 的循环并重写为 plan IR。
struct HexagonDoubleBufferPlanRewritePass
    : public ::impl::HexagonDoubleBufferPlanRewriteBase<
          HexagonDoubleBufferPlanRewritePass> {
  // 声明本 pass 会创建或依赖的 dialect。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  // 在当前 FunctionOpInterface 上执行 plan rewrite。
  void runOnOperation() override {
    // 获取当前函数。
    auto func = getOperation();
    // 获取别名分析，供 tile 数据流和合法性验证使用。
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    // 每个成功重写的计划使用递增 id。
    int64_t nextPlanId = 0;

    // 先收集所有循环，避免改写时影响 walk。
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    // 逐个尝试重写已标注的循环。
    for (scf::ForOp loop : loops) {
      // 只处理 annotate pass 标记为 pointwise 或 reduce 的循环。
      auto kind = loop->getAttrOfType<StringAttr>(kLoopKindAttr);
      if (!kind || (kind.getValue() != kPointwise && kind.getValue() != kReduce))
        continue;

      // 初始化计划并记录是否是 reduction。
      Plan plan;
      plan.isReduction = kind.getValue() == kReduce;
      // 构建 tile 数据流计划；失败则跳过该循环。
      TileDataFlowAnalysis analysis(loop, aliasAnalysis);
      if (!analysis.run(plan))
        continue;

      // 对合法计划执行 IR 重写。
      IRRewriter rewriter(loop.getContext());
      rewritePointwisePlan(rewriter, plan, nextPlanId++);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferPlanRewritePass() {
  // 返回 pass 实例，供 pipeline 注册和创建。
  return std::make_unique<HexagonDoubleBufferPlanRewritePass>();
}

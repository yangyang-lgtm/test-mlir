//===- ScheduleDoubleBufferCopiesPass.cpp --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Transforms/CopyDirection.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_SCHEDULEDOUBLEBUFFERCOPIES
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// 检查 memref.copy 是否带有期望的 copy direction 属性。
bool hasCopyDirection(memref::CopyOp copy, StringRef expected) {
  // direction 属性由前置 pass 标注。
  auto direction = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  // 属性存在且字符串匹配才算命中。
  return direction && direction.getValue() == expected;
}

// global -> shared 的 copy 视为 load。
bool isLoadCopy(memref::CopyOp copy) {
  return hasCopyDirection(copy, kGlobalToShared);
}

// shared -> global 的 copy 视为 store。
bool isStoreCopy(memref::CopyOp copy) {
  return hasCopyDirection(copy, kSharedToGlobal);
}

// 判断一个操作是否是本 pass 需要调度的 load/store copy。
bool isScheduledCopy(Operation *op) {
  // 只处理 memref.copy。
  auto copy = dyn_cast<memref::CopyOp>(op);
  // load copy 和 store copy 都属于调度目标。
  return copy && (isLoadCopy(copy) || isStoreCopy(copy));
}

// 数据流节点的角色，用于打印和调度分类。
enum class ScheduleRole { Load, Store, Compute };

// 描述一个节点对某个 tile 的读写行为。
struct DataFlowAccess {
  // tileRoots 中的索引。
  unsigned tileIndex = 0;
  // 当前节点是否读取该 tile。
  bool reads = false;
  // 当前节点是否写入该 tile。
  bool writes = false;
};

// 描述两个节点之间的数据依赖边。
struct DataFlowEdge {
  // 前驱节点在 nodes 中的索引。
  unsigned predecessor = 0;
  // 依赖发生在哪个 tile 上。
  unsigned tileIndex = 0;
  // read-after-write 依赖。
  bool raw = false;
  // write-after-read 依赖。
  bool war = false;
  // write-after-write 依赖。
  bool waw = false;
};

// 数据流图中的一个操作节点。
struct DataFlowNode {
  // 原始 MLIR operation。
  Operation *op = nullptr;
  // 节点角色：load/store/compute。
  ScheduleRole role = ScheduleRole::Compute;
  // 该操作访问的 tile 集合。
  SmallVector<DataFlowAccess> accesses;
  // 当前节点依赖的前驱边。
  SmallVector<DataFlowEdge> predecessors;
  // 是否存在无法精确归属到 value 的内存副作用。
  bool hasUnknownMemoryEffect = false;
};

// 沿着 subview/cast/view 链找到用于别名判断的根 memref。
Value findMemoryRoot(Value value) {
  // 不断剥离只改变视图或类型的操作。
  while (value) {
    // block argument 或外部值没有 defining op，直接作为根返回。
    Operation *def = value.getDefiningOp();
    if (!def)
      return value;

    // subview 的根来自 source。
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    // memref.cast 不改变底层存储。
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    // reinterpret_cast 也继续追踪 source。
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      continue;
    }
    // view 的底层存储来自 source。
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      value = view.getSource();
      continue;
    }

    // 遇到其它定义操作时停止，当前 value 作为根。
    return value;
  }

  // 输入为空时返回空 value。
  return {};
}

// 将角色枚举转换为调试输出字符串。
StringRef stringifyRole(ScheduleRole role) {
  switch (role) {
  case ScheduleRole::Load:
    return "LOAD";
  case ScheduleRole::Store:
    return "STORE";
  case ScheduleRole::Compute:
    return "COMPUTE";
  }
  llvm_unreachable("unknown schedule role");
}

// 将读写布尔组合转换为调试输出字符串。
StringRef stringifyAccess(bool reads, bool writes) {
  if (reads && writes)
    return "READ_WRITE";
  if (reads)
    return "READ";
  if (writes)
    return "WRITE";
  return "NO_ACCESS";
}

// 向访问列表中加入或合并某个 tile 的读写信息。
void addAccess(SmallVectorImpl<DataFlowAccess> &accesses, unsigned tileIndex,
               bool reads, bool writes) {
  // 如果已有同 tile 记录，就把读写标记 OR 进去。
  for (DataFlowAccess &access : accesses) {
    if (access.tileIndex == tileIndex) {
      access.reads |= reads;
      access.writes |= writes;
      return;
    }
  }

  // 该 tile 首次出现，追加一条访问记录。
  accesses.push_back(DataFlowAccess{tileIndex, reads, writes});
}

// 分析一个 block 内 load/store copy 与 compute 对 tile 的数据依赖。
class CopyScheduleDataFlowAnalysis {
public:
  // 保存待分析 block 和别名分析结果。
  CopyScheduleDataFlowAnalysis(Block &block, AliasAnalysis &aliasAnalysis)
      : block(block), aliasAnalysis(aliasAnalysis) {}

  // 构建 tile 集合、节点列表和依赖边。
  void build() {
    // 先从 copy 中找出 shared tile 根。
    collectTiles();
    // 清空旧分析结果。
    nodes.clear();
    opToNode.clear();
    // 按 block 顺序收集所有相关操作节点。
    for (Operation &op : block.without_terminator()) {
      std::optional<DataFlowNode> node = collectNode(&op);
      if (!node)
        continue;

      // 保存节点并建立 operation 到节点索引的映射。
      nodes.push_back(std::move(*node));
      opToNode.try_emplace(nodes.back().op, nodes.size() - 1);
    }
    // 根据读写顺序建立 RAW/WAR/WAW 依赖。
    buildDependencies();
  }

  // 返回当前分析中所有 load copy。
  SmallVector<memref::CopyOp> getLoads() const {
    SmallVector<memref::CopyOp> copies;
    for (const DataFlowNode &node : nodes)
      if (node.role == ScheduleRole::Load)
        copies.push_back(cast<memref::CopyOp>(node.op));
    return copies;
  }

  // 返回当前分析中所有 store copy。
  SmallVector<memref::CopyOp> getStores() const {
    SmallVector<memref::CopyOp> copies;
    for (const DataFlowNode &node : nodes)
      if (node.role == ScheduleRole::Store)
        copies.push_back(cast<memref::CopyOp>(node.op));
    return copies;
  }

  // 查询某个 operation 对应的数据流节点索引。
  std::optional<unsigned> getNodeIndex(Operation *op) const {
    auto it = opToNode.find(op);
    if (it == opToNode.end())
      return std::nullopt;
    return it->second;
  }

  // 判断当前数据流图是否为空。
  bool empty() const { return nodes.empty(); }

  // 打印数据流图，帮助观察调度前后的依赖关系。
  void print(raw_ostream &os, StringRef title) const {
    os << "\n=== ScheduleDoubleBufferCopies dataflow: " << title << " ===\n";
    os << "loop: ";
    if (Operation *parent = block.getParentOp())
      parent->print(os, OpPrintingFlags().skipRegions());
    else
      os << "<unknown>";
    os << "\n";

    os << "nodes:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      os << "  [" << nodeIndex << "] " << stringifyRole(node.role);
      if (node.hasUnknownMemoryEffect)
        os << " unknown-memory-effect";
      os << " accesses={";
      if (node.accesses.empty())
        os << "}";
      else {
        llvm::interleaveComma(node.accesses, os,
                              [&](const DataFlowAccess &access) {
                                os << "tile#" << access.tileIndex << ":"
                                   << stringifyAccess(access.reads,
                                                      access.writes);
                              });
        os << "}";
      }
      os << "\n";

      if (auto copy = dyn_cast<memref::CopyOp>(node.op)) {
        os << "      copy memory: src=" << findMemoryRoot(copy.getSource())
           << ", dst=" << findMemoryRoot(copy.getTarget()) << "\n";
      }
      os << "      op: ";
      node.op->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }

    os << "dataflow:\n";
    bool printedDependency = false;
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      for (const DataFlowEdge &edge : node.predecessors) {
        printedDependency = true;
        const DataFlowNode &predecessor = nodes[edge.predecessor];
        os << "  [" << edge.predecessor << "] "
           << stringifyRole(predecessor.role) << " --"
           << describeDependency(edge) << "(tile#" << edge.tileIndex
           << ")--> [" << nodeIndex
           << "] " << stringifyRole(node.role) << "\n";
      }
    }
    if (!printedDependency)
      os << "  <none>\n";
    os << "=== End ScheduleDoubleBufferCopies dataflow ===\n";
  }

private:
  // 从 block 中的 load/store copy 收集 shared tile 根。
  void collectTiles() {
    // 每次构建前先清空 tile 根列表。
    tileRoots.clear();
    // 扫描 block 内所有非 terminator 操作。
    for (Operation &op : block.without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy)
        continue;

      // load 的 tile 是 target，store 的 tile 是 source。
      Value tile = {};
      if (isLoadCopy(copy))
        tile = findMemoryRoot(copy.getTarget());
      else if (isStoreCopy(copy))
        tile = findMemoryRoot(copy.getSource());
      else
        continue;

      // 只把由 memref.alloc 创建的本地 tile 作为调度对象。
      if (!tile || !tile.getDefiningOp<memref::AllocOp>())
        continue;
      // 避免重复记录同一个 tile。
      if (!llvm::is_contained(tileRoots, tile))
        tileRoots.push_back(tile);
    }
  }

  // 找到某个 value 对应的 tile 索引。
  std::optional<unsigned> getTileIndex(Value value) const {
    // 先规约到内存根。
    value = findMemoryRoot(value);
    if (!value)
      return std::nullopt;
    // 用相等或 aliasAnalysis 判断它是否引用某个 tile。
    for (auto [index, tile] : llvm::enumerate(tileRoots))
      if (value == tile || aliasAnalysis.alias(value, tile))
        return index;
    // 没有匹配任何 tile。
    return std::nullopt;
  }

  // 将一个 operation 转换成数据流节点；无关操作返回 nullopt。
  std::optional<DataFlowNode> collectNode(Operation *op) const {
    // 初始化节点，默认视为 compute。
    DataFlowNode node;
    node.op = op;

    // memref.copy 根据 direction 分类为 load/store。
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (isLoadCopy(copy))
        node.role = ScheduleRole::Load;
      else if (isStoreCopy(copy))
        node.role = ScheduleRole::Store;
      else
        return std::nullopt;

      // copy source 命中 tile 表示读取该 tile。
      if (auto source = getTileIndex(copy.getSource()))
        addAccess(node.accesses, *source, /*reads=*/true, /*writes=*/false);
      // copy target 命中 tile 表示写入该 tile。
      if (auto target = getTileIndex(copy.getTarget()))
        addAccess(node.accesses, *target, /*reads=*/false, /*writes=*/true);
      return node;
    }

    // 非 copy 操作需要通过 MemoryEffectOpInterface 获取读写副作用。
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return std::nullopt;

    // 收集操作声明的内存 effects。
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      // 只关心 Read/Write effect。
      bool reads = isa<MemoryEffects::Read>(effect.getEffect());
      bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (!reads && !writes)
        continue;

      // 没有关联 value 的 effect 无法精确判断 tile，标记为 unknown。
      Value value = effect.getValue();
      if (!value) {
        node.hasUnknownMemoryEffect = true;
        continue;
      }
      // 如果 effect value 命中 tile，就记录对应读写。
      if (auto tileIndex = getTileIndex(value))
        addAccess(node.accesses, *tileIndex, reads, writes);
    }

    // 没有 tile 访问且没有未知 effect 的操作与调度无关。
    if (node.accesses.empty() && !node.hasUnknownMemoryEffect)
      return std::nullopt;
    // 返回有效数据流节点。
    return node;
  }

  // 按顺序扫描节点并建立每个节点的前驱依赖。
  void buildDependencies() {
    // 每个 tile 最近一次写的节点索引。
    SmallVector<std::optional<unsigned>> lastWrite(tileRoots.size());
    // 每个 tile 最近一次写之后发生的读节点。
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tileRoots.size());

    // 顺序处理每个节点。
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      // 当前节点的依赖边先临时收集，最后排序。
      SmallVector<DataFlowEdge> edges;
      for (const DataFlowAccess &access : node.accesses) {
        // 取出当前节点对这个 tile 的访问方式。
        bool reads = access.reads;
        bool writes = access.writes;
        unsigned tileIndex = access.tileIndex;

        // 当前读依赖上一次写，形成 RAW。
        if (reads && lastWrite[tileIndex])
          addOrMergeEdge(edges, *lastWrite[tileIndex], tileIndex,
                         /*raw=*/true, /*war=*/false, /*waw=*/false);

        // 当前写需要依赖之前的写和写后到当前写之间的读。
        if (writes) {
          // 写依赖上一次写，形成 WAW。
          if (lastWrite[tileIndex])
            addOrMergeEdge(edges, *lastWrite[tileIndex], tileIndex,
                           /*raw=*/false, /*war=*/false, /*waw=*/true);
          // 写依赖上一次写之后的读，形成 WAR。
          for (unsigned read : readsSinceWrite[tileIndex])
            addOrMergeEdge(edges, read, tileIndex, /*raw=*/false,
                           /*war=*/true, /*waw=*/false);
          // 当前写刷新读集合和 lastWrite。
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
          if (reads)
            readsSinceWrite[tileIndex].push_back(nodeIndex);
        } else if (reads) {
          // 纯读记录到 readsSinceWrite，供后续写建立 WAR。
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      // 依赖边排序，保证调试输出和行为稳定。
      llvm::sort(edges, [](const DataFlowEdge &lhs, const DataFlowEdge &rhs) {
        if (lhs.predecessor != rhs.predecessor)
          return lhs.predecessor < rhs.predecessor;
        return lhs.tileIndex < rhs.tileIndex;
      });
      // 写回当前节点的前驱列表。
      node.predecessors = std::move(edges);
    }
  }

  // 向边集合加入一条边；若已有同 predecessor/tile，则合并依赖类型。
  void addOrMergeEdge(SmallVectorImpl<DataFlowEdge> &edges,
                      unsigned predecessor, unsigned tileIndex, bool raw,
                      bool war, bool waw) const {
    for (DataFlowEdge &edge : edges) {
      if (edge.predecessor == predecessor && edge.tileIndex == tileIndex) {
        edge.raw |= raw;
        edge.war |= war;
        edge.waw |= waw;
        return;
      }
    }
    edges.push_back(DataFlowEdge{predecessor, tileIndex, raw, war, waw});
  }

  // 把依赖边类型拼成调试字符串。
  std::string describeDependency(const DataFlowEdge &edge) const {
    SmallVector<StringRef> kinds;
    if (edge.raw)
      kinds.push_back("RAW");
    if (edge.war)
      kinds.push_back("WAR");
    if (edge.waw)
      kinds.push_back("WAW");

    if (kinds.empty())
      return "UNKNOWN";

    std::string result;
    llvm::raw_string_ostream os(result);
    llvm::interleave(kinds, os, "+");
    return result;
  }

  // 被分析的 block。
  Block &block;
  // MLIR alias analysis，用于判断不同 view 是否可能指向同一 tile。
  AliasAnalysis &aliasAnalysis;
  // 当前 block 中可调度的 tile 根。
  SmallVector<Value> tileRoots;
  // 当前 block 的数据流节点。
  SmallVector<DataFlowNode> nodes;
  // operation 到节点索引的映射。
  llvm::DenseMap<Operation *, unsigned> opToNode;
};

// 判断地址计算相关操作是否可以和 copy 一起移动。
bool isMovableAddressOp(Operation *op) {
  // 不能移动带 region 的复杂操作，只允许无副作用或 memref 视图类操作。
  return op->getNumRegions() == 0 &&
         (isMemoryEffectFree(op) ||
          isa<memref::AllocOp, memref::SubViewOp, memref::ViewOp,
              memref::CastOp, memref::ReinterpretCastOp>(op));
}

// 收集 copy 依赖的、位于 anchor 和 copy 之间且可以一起移动的定义操作。
bool collectMovableDefs(Value value, Block *block, Operation *anchor,
                        Operation *copy,
                        llvm::SetVector<Operation *> &movableOps) {
  // 外部定义、其它 block 定义、或 anchor 之前定义的值不需要移动。
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != block || def->isBeforeInBlock(anchor))
    return true;
  // 定义必须位于 anchor 之后 copy 之前，且自身可移动。
  if (def == anchor || def == copy || !def->isBeforeInBlock(copy) ||
      !isMovableAddressOp(def))
    return false;

  // 递归收集定义操作的操作数依赖。
  for (Value operand : def->getOperands())
    if (!collectMovableDefs(operand, block, anchor, copy, movableOps))
      return false;

  // 依赖先被加入，SetVector 保持去重和确定顺序。
  movableOps.insert(def);
  return true;
}

// 判断 copy 跨过某个 op 时是否会破坏内存语义。
bool hasConflictWhenCrossing(Operation *op, Value source, Value target,
                             AliasAnalysis &aliasAnalysis) {
  // 不允许跨过其它被调度的 copy，避免打乱 copy 间显式顺序。
  if (isScheduledCopy(op))
    return true;

  // source/target 规约到根，用于别名判断。
  source = findMemoryRoot(source);
  target = findMemoryRoot(target);

  // 无副作用操作不会与 copy 冲突。
  if (isMemoryEffectFree(op))
    return false;

  // 不支持精确 effect 的操作保守认为有冲突。
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return true;

  // 检查每个 Read/Write effect 是否碰到 copy 的 source/target。
  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    // 只关心读写 effect。
    bool reads = isa<MemoryEffects::Read>(effect.getEffect());
    bool writes = isa<MemoryEffects::Write>(effect.getEffect());
    if (!reads && !writes)
      continue;

    // 没有 value 的 effect 无法判断，保守阻止移动。
    Value effectValue = effect.getValue();
    if (!effectValue)
      return true;
    effectValue = findMemoryRoot(effectValue);

    // 写 source 会改变 copy 读取的数据，不能跨过。
    if (writes &&
        (effectValue == source || aliasAnalysis.alias(effectValue, source)))
      return true;
    // 读或写 target 会与 copy 写入目标冲突，不能跨过。
    if ((reads || writes) &&
        (effectValue == target || aliasAnalysis.alias(effectValue, target)))
      return true;
  }

  // 没发现冲突，可以跨过。
  return false;
}

// 尝试把 load copy 提到 anchor 之前。
bool hoistLoadBefore(memref::CopyOp copy, Operation *anchor,
                     AliasAnalysis &aliasAnalysis) {
  // anchor 必须和 copy 在同一 block，且位于 copy 之前。
  if (copy == anchor || anchor->getBlock() != copy->getBlock() ||
      !anchor->isBeforeInBlock(copy))
    return false;

  // 收集 source/target 地址计算所需、需要一起提前的定义操作。
  Block *block = copy->getBlock();
  llvm::SetVector<Operation *> movableOps;
  if (!collectMovableDefs(copy.getSource(), block, anchor, copy, movableOps) ||
      !collectMovableDefs(copy.getTarget(), block, anchor, copy, movableOps))
    return false;

  // 检查 copy 从原位置移动到 anchor 前会跨过的所有操作。
  for (Operation *current = anchor; current && current != copy;
       current = current->getNextNode()) {
    // 这些地址计算操作会和 copy 一起移动，不算跨越冲突。
    if (movableOps.contains(current))
      continue;
    // 遇到内存冲突就不能提前。
    if (hasConflictWhenCrossing(current, copy.getSource(), copy.getTarget(),
                                aliasAnalysis))
      return false;
  }

  // 先移动地址计算依赖，再移动 copy 本身。
  for (Operation *op : movableOps)
    op->moveBefore(anchor);
  copy->moveBefore(anchor);
  return true;
}

// 从 block 开头开始尝试，将 load copy 提到最早合法位置。
bool hoistLoadAsEarlyAsPossible(memref::CopyOp copy,
                                AliasAnalysis &aliasAnalysis) {
  // 先快照 copy 之前的所有潜在 anchor，避免移动时迭代器失效。
  SmallVector<Operation *> anchors;
  for (Operation *op = &copy->getBlock()->front(); op && op != copy;
       op = op->getNextNode())
    anchors.push_back(op);

  // 找到第一个可行 anchor 就完成最早提前。
  for (Operation *anchor : anchors)
    if (anchor->getBlock() == copy->getBlock() &&
        anchor->isBeforeInBlock(copy) &&
        hoistLoadBefore(copy, anchor, aliasAnalysis))
      return true;

  // 没有找到合法提前位置。
  return false;
}

// 尝试把 store copy 下沉到 insertAfter 之后。
bool sinkStoreAfter(memref::CopyOp copy, Operation *insertAfter,
                    AliasAnalysis &aliasAnalysis) {
  // insertAfter 必须和 copy 在同一 block，且位于 copy 之后。
  if (copy == insertAfter || insertAfter->getBlock() != copy->getBlock() ||
      !copy->isBeforeInBlock(insertAfter))
    return false;

  // 检查从 copy 之后到 insertAfter 之前的跨越操作。
  for (Operation *current = copy->getNextNode(); current && current != nullptr;
       current = current->getNextNode()) {
    if (current == insertAfter)
      break;
    // 任一内存冲突都会阻止下沉。
    if (hasConflictWhenCrossing(current, copy.getSource(), copy.getTarget(),
                                aliasAnalysis))
      return false;
  }
  // store 下沉到 insertAfter 之后也等价于跨过 insertAfter 本身。
  if (hasConflictWhenCrossing(insertAfter, copy.getSource(), copy.getTarget(),
                              aliasAnalysis))
    return false;

  // 将 copy 移动到 insertAfter 的下一个节点之前。
  copy->moveBefore(insertAfter->getNextNode());
  return true;
}

// 尽量把 store copy 下沉到 block 末尾前的最后合法位置。
bool sinkStoreAsLateAsPossible(memref::CopyOp copy,
                               AliasAnalysis &aliasAnalysis) {
  // terminator 不能被跨过，store 最多下沉到 terminator 之前。
  Operation *terminator = copy->getBlock()->getTerminator();
  // 记录目前发现的最后一个可跨越操作。
  Operation *lastLegal = nullptr;
  for (Operation *op = copy->getNextNode(); op && op != terminator;
       op = op->getNextNode()) {
    // 遇到冲突后停止，不能继续下沉。
    if (hasConflictWhenCrossing(op, copy.getSource(), copy.getTarget(),
                                aliasAnalysis))
      break;
    lastLegal = op;
  }

  // 没有可跨越操作则无需移动。
  if (!lastLegal)
    return false;
  // 真正执行移动。
  return sinkStoreAfter(copy, lastLegal, aliasAnalysis);
}

// 对一个循环体 block 执行 copy 调度：load 尽量提前，store 尽量延后。
bool scheduleBlock(Block &block, AliasAnalysis &aliasAnalysis) {
  // 没有 terminator 的 block 不是正常可调度 block。
  if (!block.getTerminator())
    return false;

  // changed 表示整个 block 是否发生变化。
  bool changed = false;
  // localChanged 控制迭代，直到一轮没有任何移动。
  bool localChanged = true;
  // 打印调度前的数据流图。
  CopyScheduleDataFlowAnalysis before(block, aliasAnalysis);
  before.build();
  before.print(llvm::outs(), "BEFORE");

  // 反复调度，直到 load/store 都无法继续移动。
  while (localChanged) {
    localChanged = false;

    // 每轮先重新构建数据流，反映上一轮移动后的 IR。
    CopyScheduleDataFlowAnalysis analysis(block, aliasAnalysis);
    analysis.build();

    // load copy 按当前顺序尽量提前。
    for (memref::CopyOp copy : analysis.getLoads()) {
      if (copy->getBlock() != &block)
        continue;
      if (hoistLoadAsEarlyAsPossible(copy, aliasAnalysis)) {
        localChanged = true;
        changed = true;
      }
    }

    // load 移动后重新构建数据流，再处理 store。
    analysis.build();

    // store copy 反向遍历，避免先移动前面的 store 影响后面的下沉机会。
    for (memref::CopyOp copy : llvm::reverse(analysis.getStores())) {
      if (copy->getBlock() != &block)
        continue;
      if (sinkStoreAsLateAsPossible(copy, aliasAnalysis)) {
        localChanged = true;
        changed = true;
      }
    }
  }

  // 打印调度后的数据流图。
  CopyScheduleDataFlowAnalysis after(block, aliasAnalysis);
  after.build();
  after.print(llvm::outs(), "AFTER");

  // 返回该 block 是否发生了任何移动。
  return changed;
}

// Pass 主体：对已标注循环执行 double-buffer copy 调度。
struct ScheduleDoubleBufferCopiesPass
    : public ::impl::ScheduleDoubleBufferCopiesBase<
          ScheduleDoubleBufferCopiesPass> {
  // 声明依赖的 dialect。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }

  // 在当前 func::FuncOp 上运行。
  void runOnOperation() override {
    // 获取 MLIR AliasAnalysis，用于判断移动 copy 是否安全。
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    // 先收集带 loop kind 属性的循环，避免 walk 中改动 IR。
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kLoopKindAttr))
        loops.push_back(loop);
    });

    // 对每个目标循环体单独调度。
    for (scf::ForOp loop : loops)
      scheduleBlock(*loop.getBody(), aliasAnalysis);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createScheduleDoubleBufferCopiesPass() {
  // 返回 pass 实例，供 pipeline 注册和创建。
  return std::make_unique<ScheduleDoubleBufferCopiesPass>();
}

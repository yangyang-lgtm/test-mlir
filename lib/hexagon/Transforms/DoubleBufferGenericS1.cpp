//===- DoubleBufferGenericS1.cpp - Double Buffer Generic Pass : Stage 1 ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass transforms single-buffered tiled linalg-generic loops into
// software-pipelined double-buffered loops using two buffer sets to overlap DMA
// transfers with computation. It creates a prologue to prefetch the first
// iteration, then carries the current-buffer selector through one loop. Each
// iteration prefetches the next tile, waits for and computes the current tile,
// writes it back, and flips the selector.
//
// This implementation is mainly in two main passes. First is this one and
// the next is in DoubleBufferGenericS2.cpp
//
// Future extensions will support additional cases including >2D DMA transfers,
// reduction loops, multi-buffering strategies, and broader pattern coverage.
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/CopyDirection.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "double-buffer-generic-s1"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERGENERICS1
#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERREDUCES1
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Structs to parse and store the schedule.
struct ScheduleTriplet {
  memref::AllocOp alloc;
  memref::CopyOp load;
};

struct SingleBufferSchedule {
  scf::ForOp forOp;
  SmallVector<ScheduleTriplet> triplets;
  SmallVector<Operation *> computeOps;
  SmallVector<memref::CopyOp> stores;
  bool isReduction = false;
};

/// Return one tile-sized view into a backing allocation whose leading
/// dimension contains both physical double-buffer slots.
Value createDoubleBufferView(IRRewriter &rewriter, Location loc, Value backing,
                             MemRefType tileType, Value leadingOffset) {
  int64_t rank = tileType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  offsets.front() = leadingOffset;
  for (int64_t size : tileType.getShape())
    sizes.push_back(rewriter.getIndexAttr(size));

  auto backingType = cast<MemRefType>(backing.getType());
  auto viewType = memref::SubViewOp::inferResultType(
      backingType, offsets, sizes, strides);
  return memref::SubViewOp::create(rewriter, loc, viewType, backing, offsets,
                                   sizes, strides);
}

/// 检查 memref.copy 是否具有预期的数据搬运方向。
bool hasCopyDirection(memref::CopyOp copy, StringRef expected) {
  auto direction = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return direction && direction.getValue() == expected;
}

memref::AllocOp findBaseAlloc(Value value) {
  while (true) {
    if (auto alloc = value.getDefiningOp<memref::AllocOp>())
      return alloc;
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      value = reinterpret.getSource();
      continue;
    }
    return nullptr;
  }
}

bool isAvailableBeforeCopy(memref::AllocOp alloc, scf::ForOp forOp,
                           memref::CopyOp copy) {
  Block *forBody = forOp.getBody();
  if (alloc->getBlock() == forBody)
    return alloc->isBeforeInBlock(copy);
  return alloc->getBlock() == forOp->getBlock() &&
         alloc->isBeforeInBlock(forOp);
}

bool isDefinedInBlockBefore(Operation *op, Block *block, Operation *limit) {
  return op && op->getBlock() == block && op->isBeforeInBlock(limit);
}

Value cloneValueSlice(Value value, IRRewriter &rewriter, IRMapping &mapping,
                      Block *sourceBlock, Operation *limit) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  auto *def = value.getDefiningOp();
  if (!isDefinedInBlockBefore(def, sourceBlock, limit))
    return value;

  for (Value operand : def->getOperands())
    cloneValueSlice(operand, rewriter, mapping, sourceBlock, limit);

  Operation *cloned = def->clone(mapping);
  // A mapped source may have a different strided offset (notably when it is a
  // dynamically selected view into the combined double buffer). Re-infer a
  // cloned subview's layout instead of retaining the old source-dependent type.
  if (auto subview = dyn_cast<memref::SubViewOp>(cloned)) {
    auto sourceType = cast<MemRefType>(subview.getSource().getType());
    subview.getResult().setType(memref::SubViewOp::inferResultType(
        sourceType, subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides()));
  }
  rewriter.insert(cloned);
  mapping.map(def->getResults(), cloned->getResults());
  return mapping.lookup(value);
}

memref::CopyOp cloneCopyWithMappedSlices(memref::CopyOp copy,
                                         IRRewriter &rewriter,
                                         IRMapping &mapping, Block *sourceBlock,
                                         StringRef role = {}) {
  Value source =
      cloneValueSlice(copy.getSource(), rewriter, mapping, sourceBlock, copy);
  Value target =
      cloneValueSlice(copy.getTarget(), rewriter, mapping, sourceBlock, copy);
  auto clonedCopy =
      memref::CopyOp::create(rewriter, copy.getLoc(), source, target);
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    clonedCopy->setAttr(kCopyDirectionAttrName, direction);
  // S1 显式记录 copy 的调度角色，避免 S2 再根据操作位置猜测。
  if (!role.empty())
    clonedCopy->setAttr("db_copy_role", rewriter.getStringAttr(role));
  return clonedCopy;
}

Operation *cloneOpWithMappedSlices(Operation *op, IRRewriter &rewriter,
                                   IRMapping &mapping, Block *sourceBlock) {
  for (Value operand : op->getOperands())
    cloneValueSlice(operand, rewriter, mapping, sourceBlock, op);

  Operation *cloned = op->clone(mapping);
  rewriter.insert(cloned);
  mapping.map(op->getResults(), cloned->getResults());
  // 标记由 S1 克隆出的计算节点，供 S2 恢复调度结构。
  cloned->setAttr("db_compute", UnitAttr::get(op->getContext()));
  return cloned;
}

/// 单个操作对 tile buffer 的访问类型，读写可以按位组合。
enum TileAccessKind : uint8_t {
  NoTileAccess = 0,
  TileRead = 1,
  TileWrite = 2,
};

struct TileAccessNode {
  Operation *op;
  // accesses[i] 表示当前操作对第 i 个 tile buffer 的读写方式。
  SmallVector<uint8_t> accesses;
  // 按循环体顶层操作顺序记录 RAW/WAR/WAW 前驱。
  SmallVector<unsigned> predecessors;
  bool isPreload = false;
  bool isCompute = false;
  bool isStore = false;
};

/// 为一个候选循环中的 tile alloc 构建局部 MemorySSA 风格的依赖图。
///
/// 每个 tile buffer 拥有独立的内存版本链。当前只处理循环体顶层的线性
/// 调度，不表达任意 CFG 汇合；遇到无法分类的 tile 访问时保守放弃转换。
struct TileMemorySSAAnalysis {
  TileMemorySSAAnalysis(AliasAnalysis &aliasAnalysis, scf::ForOp forOp,
                        SingleBufferSchedule &schedule)
      : aliasAnalysis(aliasAnalysis), forOp(forOp), schedule(schedule) {}

  AliasAnalysis &aliasAnalysis;
  scf::ForOp forOp;
  SingleBufferSchedule &schedule;
  SmallVector<memref::AllocOp> tileAllocs;
  llvm::SmallDenseMap<Operation *, unsigned> tileIndices;
  llvm::SmallDenseSet<Operation *> preloadCopies;
  SmallVector<TileAccessNode> graph;

  static StringRef getNodeRole(const TileAccessNode &node) {
    if (node.isPreload)
      return "LOAD";
    if (node.isCompute)
      return "COMPUTE";
    if (node.isStore)
      return "STORE";
    return "UNKNOWN";
  }

  static StringRef getAccessName(uint8_t access) {
    if (access == (TileRead | TileWrite))
      return "READ_WRITE";
    if (access == TileRead)
      return "READ";
    if (access == TileWrite)
      return "WRITE";
    return "NONE";
  }

  /// 打印分析得到的节点分类、tile 访问方式和内存依赖流向。
  void print(raw_ostream &os) const {
    os << "\n=== TileMemorySSAAnalysis ===\n";
    os << "loop: ";
    forOp->print(os, OpPrintingFlags().skipRegions());
    os << "\n";

    os << "tiles:\n";
    for (auto [tileIndex, alloc] : llvm::enumerate(tileAllocs)) {
      os << "  tile#" << tileIndex << ": ";
      alloc->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }

    os << "nodes:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      os << "  [" << nodeIndex << "] " << getNodeRole(node) << " accesses={";
      bool firstAccess = true;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        if (access == NoTileAccess)
          continue;
        if (!firstAccess)
          os << ", ";
        firstAccess = false;
        os << "tile#" << tileIndex << ":" << getAccessName(access);
      }
      os << "}\n";
      os << "      op: ";
      node.op->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }

    os << "dataflow:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      for (unsigned predecessor : node.predecessors) {
        const TileAccessNode &source = graph[predecessor];
        bool printedEdge = false;
        for (unsigned tileIndex = 0; tileIndex < tileAllocs.size();
             ++tileIndex) {
          uint8_t sourceAccess = source.accesses[tileIndex];
          uint8_t targetAccess = node.accesses[tileIndex];
          bool raw = (sourceAccess & TileWrite) && (targetAccess & TileRead);
          bool war = (sourceAccess & TileRead) && (targetAccess & TileWrite);
          bool waw = (sourceAccess & TileWrite) && (targetAccess & TileWrite);
          if (!raw && !war && !waw)
            continue;

          os << "  [" << predecessor << "] " << getNodeRole(source) << " --";
          if (raw)
            os << "RAW";
          if (war)
            os << (raw ? "+WAR" : "WAR");
          if (waw)
            os << ((raw || war) ? "+WAW" : "WAW");
          os << "(tile#" << tileIndex << ")--> [" << nodeIndex << "] "
             << getNodeRole(node) << "\n";
          printedEdge = true;
        }
        if (!printedEdge)
          os << "  [" << predecessor << "] " << getNodeRole(source)
             << " --DEPENDENCE--> [" << nodeIndex << "] " << getNodeRole(node)
             << "\n";
      }
    }
    os << "=== End TileMemorySSAAnalysis ===\n";
  }

  bool isAllowedLocalUtilityOp(Operation *op) {
    // 这些操作只构造或转换 memref 视图，本身不作为内存版本节点。
    if (isa<memref::AllocOp, memref::DeallocOp, memref::SubViewOp,
            memref::CastOp, memref::ReinterpretCastOp>(op))
      return true;
    return isMemoryEffectFree(op);
  }

  std::optional<unsigned> getTileIndex(Value value) {
    if (!isa<BaseMemRefType>(value.getType()))
      return std::nullopt;
    // subview/cast 等派生值通过 AliasAnalysis 归并到对应 tile alloc。
    for (auto [index, alloc] : llvm::enumerate(tileAllocs)) {
      if (aliasAnalysis.alias(value, alloc.getMemref()))
        return index;
    }
    return std::nullopt;
  }

  bool accessesAnyTile(ArrayRef<uint8_t> accesses) {
    return llvm::any_of(accesses,
                        [](uint8_t access) { return access != NoTileAccess; });
  }

  void addCopyAccesses(memref::CopyOp copy, TileAccessNode &node) {
    // copy 的 source 是读，target 是写。
    if (auto source = getTileIndex(copy.getSource()))
      node.accesses[*source] |= TileRead;
    if (auto target = getTileIndex(copy.getTarget()))
      node.accesses[*target] |= TileWrite;
  }

  bool addPreciseEffectAccesses(Operation *op, TileAccessNode &node) {
    auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffectOp)
      return false;

    SmallVector<MemoryEffects::EffectInstance> effects;
    memoryEffectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
      if (!isRead && !isWrite)
        continue;

      // 没有关联具体 Value 的读写 effect 无法精确归属到某个 tile。
      Value effectValue = effect.getValue();
      if (!effectValue)
        return false;

      for (auto [index, alloc] : llvm::enumerate(tileAllocs)) {
        if (!aliasAnalysis.alias(effectValue, alloc.getMemref()))
          continue;
        if (isRead)
          node.accesses[index] |= TileRead;
        if (isWrite)
          node.accesses[index] |= TileWrite;
      }
    }
    return true;
  }

  void addConservativeModRefAccesses(Operation *op, TileAccessNode &node) {
    // 无法获得带 Value 的精确 effect 时，回退到保守 ModRef 查询。
    for (auto [index, alloc] : llvm::enumerate(tileAllocs)) {
      ModRefResult modRef = aliasAnalysis.getModRef(op, alloc.getMemref());
      if (modRef.isRef())
        node.accesses[index] |= TileRead;
      if (modRef.isMod())
        node.accesses[index] |= TileWrite;
    }
  }

  bool collectCopyIns() {
    Block *forBody = forOp.getBody();
    for (Operation &op : forBody->without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      // 当前双缓冲模式只接受 global -> shared 的输入预取。
      if (!copy || !hasCopyDirection(copy, kGlobalToShared))
        continue;
      auto sourceAlloc = findBaseAlloc(copy.getSource());
      // tile -> tile 或 tile -> external 的 copy 不是输入预取。
      if (sourceAlloc && tileIndices.contains(sourceAlloc.getOperation()))
        continue;
      auto alloc = findBaseAlloc(copy.getTarget());
      if (!alloc || !isAvailableBeforeCopy(alloc, forOp, copy))
        continue;
      // 同一 tile 出现多个预取定义时，当前线性版本模型无法唯一分类。
      if (tileIndices.contains(alloc.getOperation()))
        return false;
      schedule.triplets.push_back({alloc, copy});
      tileIndices[alloc.getOperation()] = tileAllocs.size();
      tileAllocs.push_back(alloc);
      preloadCopies.insert(copy.getOperation());
    }

    return !schedule.triplets.empty();
  }

  bool buildAccessGraph() {
    graph.clear();
    // lastWrite 记录每个 tile 当前的版本定义者；
    // readsSinceWrite 用于在下一次写入时补充 WAR 依赖。
    SmallVector<std::optional<unsigned>> lastWrite(tileAllocs.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tileAllocs.size());

    for (Operation &op : forOp.getBody()->without_terminator()) {
      TileAccessNode node{
          &op, SmallVector<uint8_t>(tileAllocs.size(), NoTileAccess)};
      if (auto copy = dyn_cast<memref::CopyOp>(&op))
        addCopyAccesses(copy, node);
      else if (!isAllowedLocalUtilityOp(&op) &&
               !addPreciseEffectAccesses(&op, node))
        addConservativeModRefAccesses(&op, node);

      if (!accessesAnyTile(node.accesses))
        continue;

      node.isPreload = preloadCopies.contains(&op);
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        bool readsTile = getTileIndex(copy.getSource()).has_value();
        bool writesTile = getTileIndex(copy.getTarget()).has_value();
        // 结果回写必须是 shared -> global，并且只读取当前 tile。
        node.isStore =
            hasCopyDirection(copy, kSharedToGlobal) && readsTile && !writesTile;
      }

      unsigned nodeIndex = graph.size();
      llvm::SmallDenseSet<unsigned> predecessors;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        bool reads = access & TileRead;
        bool writes = access & TileWrite;
        // 读依赖最近一次写：RAW。
        if (reads && lastWrite[tileIndex])
          predecessors.insert(*lastWrite[tileIndex]);
        if (writes) {
          // 写依赖最近一次写和该版本之后的所有读：WAW + WAR。
          if (lastWrite[tileIndex])
            predecessors.insert(*lastWrite[tileIndex]);
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
        } else if (reads)
          readsSinceWrite[tileIndex].push_back(nodeIndex);
      }
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors);
      graph.push_back(std::move(node));
    }
    return classifyComputeNodes();
  }

  bool writesLoopCarriedExternalBuffer(Operation *op) {
    auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffectOp)
      return false;

    SmallVector<MemoryEffects::EffectInstance> effects;
    memoryEffectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      Value value = effect.getValue();
      if (!value || !isa<BaseMemRefType>(value.getType()) ||
          getTileIndex(value))
        continue;

      Operation *def = value.getDefiningOp();
      if (!def || def->getBlock() != forOp.getBody())
        return true;
    }
    return false;
  }

  bool classifyComputeNodes() {
    // 从所有 preload 正向传播，找出由输入预取可达的节点。
    llvm::SmallBitVector reachableFromPreload(graph.size());
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      bool reachable = node.isPreload;
      for (unsigned predecessor : node.predecessors)
        reachable |= reachableFromPreload.test(predecessor);
      if (reachable)
        reachableFromPreload.set(nodeIndex);
    }

    // 从所有 store 沿前驱边反向传播，找出能够影响结果回写的节点。
    llvm::SmallBitVector reachesStore(graph.size());
    for (auto [nodeIndex, node] : llvm::enumerate(graph))
      if (node.isStore)
        reachesStore.set(nodeIndex);
    for (int64_t nodeIndex = static_cast<int64_t>(graph.size()) - 1;
         nodeIndex >= 0; --nodeIndex) {
      if (!reachesStore.test(nodeIndex))
        continue;
      for (unsigned predecessor : graph[nodeIndex].predecessors)
        reachesStore.set(predecessor);
    }

    bool hasStore = llvm::any_of(
        graph, [](const TileAccessNode &node) { return node.isStore; });
    bool hasReductionCompute = false;
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      if (node.isPreload || node.isStore)
        continue;

      // 普通 tile 流要求 preload -> compute -> store。Reduction 流没有
      // 循环内 store，而是原地更新循环外 accumulator。
      node.isCompute = reachableFromPreload.test(nodeIndex) &&
                       (hasStore ? reachesStore.test(nodeIndex)
                                 : writesLoopCarriedExternalBuffer(node.op));
      hasReductionCompute |= !hasStore && node.isCompute;

      // copy 只能作为已识别的 preload/store，不能夹在计算数据流内部。
      if (isa<memref::CopyOp>(node.op) || !node.isCompute)
        return false;
    }
    schedule.isReduction = !hasStore && hasReductionCompute;
    return hasStore || hasReductionCompute;
  }

  bool refineCopyInsToComputeUsedBuffers() {
    // 删除没有被计算节点实际访问的预取，避免为无关 copy 分配双缓冲。
    llvm::SmallBitVector usedByCompute(tileAllocs.size());
    for (const TileAccessNode &node : graph) {
      if (!node.isCompute)
        continue;
      for (auto [index, access] : llvm::enumerate(node.accesses))
        if (access != NoTileAccess)
          usedByCompute.set(index);
    }

    SmallVector<ScheduleTriplet> refinedTriplets;
    SmallVector<memref::AllocOp> refinedAllocs;
    llvm::SmallDenseMap<Operation *, unsigned> refinedIndices;
    llvm::SmallDenseSet<Operation *> refinedPreloadCopies;
    for (auto [index, triplet] : llvm::enumerate(schedule.triplets)) {
      if (!usedByCompute.test(index))
        continue;
      refinedIndices[triplet.alloc.getOperation()] = refinedAllocs.size();
      refinedTriplets.push_back(triplet);
      refinedAllocs.push_back(triplet.alloc);
      refinedPreloadCopies.insert(triplet.load.getOperation());
    }
    if (refinedTriplets.empty())
      return false;
    schedule.triplets = std::move(refinedTriplets);
    tileAllocs = std::move(refinedAllocs);
    tileIndices = std::move(refinedIndices);
    preloadCopies = std::move(refinedPreloadCopies);
    return true;
  }

  bool collectScheduleFromGraph() {
    // DAG 已完成分类，此处按原始拓扑顺序生成改写 schedule。
    schedule.computeOps.clear();
    schedule.stores.clear();
    for (const TileAccessNode &node : graph) {
      if (node.isCompute)
        schedule.computeOps.push_back(node.op);
      if (node.isStore)
        schedule.stores.push_back(cast<memref::CopyOp>(node.op));
    }
    return !schedule.computeOps.empty() &&
           (schedule.isReduction || !schedule.stores.empty());
  }

  bool validateScheduleDependencies() {
    // 每个 tile 必须满足 preload -> compute -> store 的阶段顺序。
    llvm::SmallBitVector seenPreload(tileAllocs.size());
    llvm::SmallBitVector seenCompute(tileAllocs.size());
    llvm::SmallBitVector seenStore(tileAllocs.size());

    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      // 局部图应天然保持拓扑序；出现反向边表示构图或分类不一致。
      if (llvm::any_of(node.predecessors, [&](unsigned predecessor) {
            return predecessor >= nodeIndex;
          }))
        return false;

      for (auto [index, access] : llvm::enumerate(node.accesses)) {
        if (access == NoTileAccess)
          continue;
        if (node.isPreload) {
          if (!(access & TileWrite) || seenPreload.test(index) ||
              seenCompute.test(index) || seenStore.test(index))
            return false;
          seenPreload.set(index);
          continue;
        }
        if (node.isCompute) {
          if (!seenPreload.test(index) || seenStore.test(index))
            return false;
          seenCompute.set(index);
          continue;
        }
        if (node.isStore) {
          if (!(access & TileRead) || !seenCompute.test(index))
            return false;
          seenStore.set(index);
        }
      }
    }

    if (seenPreload.count() != tileAllocs.size())
      return false;
    if (schedule.isReduction)
      return seenCompute.count() != 0 && seenStore.none();
    // 只要求被回写的 tile 确实经过计算；纯输入 tile 不要求 store。
    for (memref::CopyOp store : schedule.stores) {
      auto tile = getTileIndex(store.getSource());
      if (!tile || !seenCompute.test(*tile))
        return false;
    }
    return true;
  }

  bool run() {
    // 第一次构图用于剔除未参与计算的预取；收缩 tile 集合后重新构图，
    // 保证最终节点索引和访问向量一致。
    if (!collectCopyIns() || !buildAccessGraph() ||
        !refineCopyInsToComputeUsedBuffers() || !buildAccessGraph() ||
        !collectScheduleFromGraph() || !validateScheduleDependencies())
      return false;

    print(llvm::outs());
    return true;
  }
};

/// Rewrite the single-buffered loop as double buffered.
void rewriteAsDoubleBuffered(IRRewriter &rewriter, scf::ForOp sbForOp,
                             SingleBufferSchedule &schedule, int &uid) {
  auto loc = sbForOp.getLoc();
  auto context = sbForOp.getContext();

  auto boolType = rewriter.getI1Type();

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sbForOp);

  // Define some general constants.
  Value trueVal = arith::ConstantOp::create(rewriter, loc, boolType,
                                            rewriter.getBoolAttr(true));
  // Allocate one backing memref per tile. Its leading dimension stores the two
  // physical slots consecutively.
  int64_t alignment = 2048;
  auto alignmentAttr = rewriter.getI64IntegerAttr(alignment);
  Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value, 3> backingBuffers;
  SmallVector<Value, 3> firstBuffers;
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    auto alloc = schedule.triplets[i].alloc;
    auto tileType = cast<MemRefType>(alloc.getType());
    SmallVector<int64_t> backingShape(tileType.getShape());
    backingShape.front() *= 2;
    auto backingType =
        MemRefType::get(backingShape, tileType.getElementType(),
                        tileType.getLayout(), tileType.getMemorySpace());
    Value backing = memref::AllocOp::create(
        rewriter, loc, backingType, mlir::ValueRange{}, alignmentAttr);
    backingBuffers.push_back(backing);
    firstBuffers.push_back(
        createDoubleBufferView(rewriter, loc, backing, tileType, zeroIndex));
  }

  // Get the bounds from the single-buffer original for-loop.
  Value lowerBound = sbForOp.getLowerBound();
  Value upperBound = sbForOp.getUpperBound();
  Value step = sbForOp.getStep();
  auto indVar = sbForOp.getInductionVar();
  auto idAttr = mlir::IntegerAttr::get(rewriter.getI64Type(), uid++);

  // Prologue: executes iff not 0-iteration loop.
  Value mayLoop =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            lowerBound, upperBound)
          .getResult();
  auto ifMayLoop =
      scf::IfOp::create(rewriter, loc, TypeRange(), mayLoop, false);
  ifMayLoop->setAttr("db_generic", idAttr);
  ifMayLoop->setAttr("db_prologue", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifMayLoop.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    IRMapping mapping;
    mapping.map(indVar, lowerBound);
    mapping.map(schedule.triplets[i].alloc, firstBuffers[i]);
    cloneCopyWithMappedSlices(schedule.triplets[i].load, rewriter, mapping,
                              sbForOp.getBody(), "prefetch");
  }

  // Kernel: carry the current-buffer selector instead of cloning ping and pong
  // sub-kernels. This keeps the compute slice single-copy.
  rewriter.setInsertionPoint(sbForOp);
  auto dbLoop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step,
                                   ValueRange{trueVal});
  dbLoop->setAttr("db_generic", idAttr);
  Block *forBody = dbLoop.getBody();
  if (!forBody->empty() && isa<scf::YieldOp>(forBody->back()))
    forBody->back().erase();
  rewriter.setInsertionPointToStart(forBody);
  Value dbIndVar = dbLoop.getInductionVar();
  Value cur = dbLoop.getRegionIterArgs().front();

  // cur=true means slot 0 is current and slot 1 is next. Convert the selector
  // to offsets instead of selecting between two memref SSA values.
  Value nextSlot =
      arith::IndexCastUIOp::create(rewriter, loc, rewriter.getIndexType(), cur);
  Value currentSlotBit =
      arith::XOrIOp::create(rewriter, loc, cur, trueVal);
  Value currentSlot = arith::IndexCastUIOp::create(
      rewriter, loc, rewriter.getIndexType(), currentSlotBit);

  SmallVector<Value, 3> currentBuffers;
  SmallVector<Value, 3> nextBuffers;
  for (auto [backing, triplet] :
       llvm::zip(backingBuffers, schedule.triplets)) {
    auto tileType = cast<MemRefType>(triplet.alloc.getType());
    Value tileExtent =
        arith::ConstantIndexOp::create(rewriter, loc, tileType.getDimSize(0));
    Value currentOffset =
        arith::MulIOp::create(rewriter, loc, currentSlot, tileExtent);
    Value nextOffset =
        arith::MulIOp::create(rewriter, loc, nextSlot, tileExtent);
    currentBuffers.push_back(createDoubleBufferView(
        rewriter, loc, backing, tileType, currentOffset));
    nextBuffers.push_back(createDoubleBufferView(rewriter, loc, backing,
                                                 tileType, nextOffset));
  }

  // Check if next preload should happen (or this is last iteration).
  Value nextIdx =
      arith::AddIOp::create(rewriter, loc, dbIndVar, step).getResult();
  Value nextExists =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            nextIdx, upperBound)
          .getResult();

  auto ifNext =
      scf::IfOp::create(rewriter, loc, TypeRange(), nextExists, false);
  ifNext->setAttr("db_prefetch", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifNext.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    IRMapping mapping;
    mapping.map(indVar, nextIdx);
    mapping.map(schedule.triplets[i].alloc, nextBuffers[i]);
    cloneCopyWithMappedSlices(schedule.triplets[i].load, rewriter, mapping,
                              sbForOp.getBody(), "prefetch");
  }

  // Compute and store the current tile after the next-tile prefetch has
  // started. Stage 2 inserts the current DMA waits at this boundary.
  rewriter.setInsertionPointAfter(ifNext);
  IRMapping mapping;
  mapping.map(indVar, dbIndVar);
  for (auto i = 0; i < schedule.triplets.size(); ++i)
    mapping.map(schedule.triplets[i].alloc, currentBuffers[i]);
  for (Operation *compute : schedule.computeOps)
    cloneOpWithMappedSlices(compute, rewriter, mapping, sbForOp.getBody());
  for (memref::CopyOp store : schedule.stores)
    cloneCopyWithMappedSlices(store, rewriter, mapping, sbForOp.getBody(),
                              "store");

  Value nextCur = arith::XOrIOp::create(rewriter, loc, cur, trueVal);
  scf::YieldOp::create(rewriter, loc, nextCur);
  rewriter.eraseOp(sbForOp);
}

// State machine to parse the `tiled_generic`.
bool generateSchedule(IRRewriter &rewriter, scf::ForOp forOp,
                      SingleBufferSchedule &schedule,
                      AliasAnalysis &aliasAnalysis, bool requireReduction) {
  if (!forOp.getInitArgs().empty())
    return false;
  schedule.forOp = forOp;

  TileMemorySSAAnalysis analysis{aliasAnalysis, forOp, schedule};
  if (!analysis.run())
    return false;
  if (schedule.isReduction != requireReduction)
    return false;

  // The combined backing allocation currently requires a non-scalar static
  // tile shape so the second slot has a compile-time leading offset.
  return llvm::all_of(schedule.triplets, [](ScheduleTriplet triplet) {
    auto type = cast<MemRefType>(triplet.alloc.getType());
    return type.getRank() > 0 && type.hasStaticShape();
  });
}

// The concrete implementation of the DoubleBufferGenericS1 pass.
struct HexagonDoubleBufferGenericS1Pass
    : public ::impl::HexagonDoubleBufferGenericS1Base<
          HexagonDoubleBufferGenericS1Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    int uniqueID = 0; // for each double-buffered linalg-generic.
    auto func = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    func.walk([&uniqueID, &aliasAnalysis](scf::ForOp forOp) {
      SingleBufferSchedule schedule;
      IRRewriter rewriter(forOp.getContext());
      bool viableCandidate = generateSchedule(
          rewriter, forOp, schedule, aliasAnalysis,
          /*requireReduction=*/false);
      if (viableCandidate)
        rewriteAsDoubleBuffered(rewriter, forOp, schedule, uniqueID);
      return WalkResult::advance();
    });
  }
};

struct HexagonDoubleBufferReduceS1Pass
    : public ::impl::HexagonDoubleBufferReduceS1Base<
          HexagonDoubleBufferReduceS1Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    int uniqueID = 0;
    auto func = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    func.walk([&](scf::ForOp forOp) {
      SingleBufferSchedule schedule;
      IRRewriter rewriter(forOp.getContext());
      if (generateSchedule(rewriter, forOp, schedule, aliasAnalysis,
                           /*requireReduction=*/true)) {
        rewriteAsDoubleBuffered(rewriter, forOp, schedule, uniqueID);
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

/// Creates an instance of the DoubleBufferGenericS1 pass.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS1Pass() {
  return std::make_unique<HexagonDoubleBufferGenericS1Pass>();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferReduceS1Pass() {
  return std::make_unique<HexagonDoubleBufferReduceS1Pass>();
}

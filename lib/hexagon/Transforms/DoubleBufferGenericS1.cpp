//===- DoubleBufferGenericS1.cpp - Double Buffer Generic Pass : Stage 1 ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// 本 pass 将单缓冲 tiled linalg-generic loop 改写为软件流水双缓冲 loop，
// 用两组物理 buffer 让 DMA 传输和计算重叠。它会创建 prologue 预取第一轮，
// 然后在一个 loop 中携带 current-buffer selector。每轮先预取下一块 tile，
// 再等待并计算当前 tile、写回结果，最后翻转 selector。
//
// 双缓冲主要分两阶段实现：本文件是 S1，DoubleBufferGenericS2.cpp 是 S2。
//
// 后续可扩展支持更多场景，例如更高维 DMA、多缓冲策略和更广的模式覆盖。
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
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// 用于解析并保存单缓冲调度的结构体。
struct ScheduleTriplet {
  // 原始单缓冲循环中的 shared/tile buffer root，通常是 alloc 或 view。
  Value tile;
  // 输入预取 copy 前的 tile 初始化/清理操作，例如 masked load 的 padding fill。
  SmallVector<Operation *> setupOps;
  // 将 global 数据拷贝到该 tile buffer 的输入预取 copy。
  memref::CopyOp load;
};

struct SingleBufferSchedule {
  // 被识别并准备改写的原始 scf.for。
  scf::ForOp forOp;
  // 每个输入 tile 的 alloc 和 load copy。
  SmallVector<ScheduleTriplet> triplets;
  // 被分类为 compute 的原始循环体操作，改写时会克隆到新循环中。
  SmallVector<Operation *> computeOps;
  // 普通 generic 形态下 shared -> global 的结果回写 copy。
  SmallVector<memref::CopyOp> stores;
  // true 表示 reduction 形态：没有逐 tile store，而是 compute 写外部 accumulator。
  bool isReduction = false;
};

/// 在 backing allocation 中返回一个 tile 大小的 view；backing 第一维包含两个物理 slot。
Value createDoubleBufferView(IRRewriter &rewriter, Location loc, Value backing,
                             MemRefType tileType, Value leadingOffset) {
  // subview 的 rank 与原 tile rank 一致。
  int64_t rank = tileType.getRank();
  // 默认所有维度 offset 为 0，后面只改第一维以选择 ping/pong slot。
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  // sizes 使用原 tile 的完整形状。
  SmallVector<OpFoldResult> sizes;
  // 这里创建连续 tile view，所以所有 stride 都是 1。
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // 第一维 offset 决定当前 view 指向 backing buffer 的哪个物理 slot。
  offsets.front() = leadingOffset;
  // 将原 tile 的每个静态维度写入 subview sizes。
  for (int64_t size : tileType.getShape())
    sizes.push_back(rewriter.getIndexAttr(size));

  // backing 是第一维扩大 2 倍后的真实分配。
  auto backingType = cast<MemRefType>(backing.getType());
  // 根据 remapped backing 类型重新推导 subview 类型，避免沿用旧 tile layout。
  auto viewType = memref::SubViewOp::inferResultType(
      backingType, offsets, sizes, strides);
  // 返回一个 tile 大小的 memref.subview，后续 copy/compute 都使用这个 view。
  return memref::SubViewOp::create(rewriter, loc, viewType, backing, offsets,
                                   sizes, strides);
}

/// 检查 memref.copy 是否具有预期的数据搬运方向。
bool hasCopyDirection(memref::CopyOp copy, StringRef expected) {
  // copy_direction 由前置 AnnotateMemrefCopyDirection pass 写入。
  auto direction = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  // 没有方向属性或方向不匹配时，该 copy 不属于当前调度角色。
  return direction && direction.getValue() == expected;
}

Value findTileRoot(Value value) {
  // 沿着 view/cast 链一直向源头追溯。
  while (true) {
    // 找到 memref.alloc 就认为找到了 tile 的底层分配。
    if (auto alloc = value.getDefiningOp<memref::AllocOp>())
      return alloc.getMemref();
    // memref.view 通常是共享 i8 slab 上切出的 typed tile，保留 typed view
    // 作为可替换的 tile root。
    if (auto view = value.getDefiningOp<memref::ViewOp>())
      return view.getResult();
    // subview 不创建新存储，继续追溯它的 source。
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    // cast 不改变底层存储，继续追溯 source。
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    // reinterpret_cast 也视为同一底层存储的不同视图。
    if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      value = reinterpret.getSource();
      continue;
    }
    // 其它来源无法静态确认是可双缓冲 tile alloc，保守返回空。
    return {};
  }
}

bool isAvailableBeforeCopy(Value tile, scf::ForOp forOp, memref::CopyOp copy) {
  Operation *def = tile.getDefiningOp();
  if (!def)
    return false;
  // tile root 如果在 loop body 内，必须支配对应 copy。
  Block *forBody = forOp.getBody();
  if (def->getBlock() == forBody)
    return def->isBeforeInBlock(copy);
  // tile root 如果在 loop 外，必须位于 forOp 之前，才能被新 prologue/kernel 使用。
  return def->getBlock() == forOp->getBlock() && def->isBeforeInBlock(forOp);
}

bool isDefinedInBlockBefore(Operation *op, Block *block, Operation *limit) {
  // cloneValueSlice 只克隆同一 block 内、位于待克隆操作之前的局部定义。
  return op && op->getBlock() == block && op->isBeforeInBlock(limit);
}

Value cloneValueSlice(Value value, IRRewriter &rewriter, IRMapping &mapping,
                      Block *sourceBlock, Operation *limit) {
  // 如果这个 value 已经被映射到新 IR 中，直接复用，避免重复克隆。
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  // 取得 value 的定义操作；block argument 或外部 value 没有 defining op。
  auto *def = value.getDefiningOp();
  // 只有原 loop body 中、且位于 limit 之前的局部地址计算才需要克隆。
  if (!isDefinedInBlockBefore(def, sourceBlock, limit))
    return value;

  // 先递归克隆 operand 的定义，保证新 IR 中 SSA dominance 正确。
  for (Value operand : def->getOperands())
    cloneValueSlice(operand, rewriter, mapping, sourceBlock, limit);

  // 使用当前 mapping 克隆定义操作；被映射的 tile alloc 会替换成 current/next view。
  Operation *cloned = def->clone(mapping);
  // 映射后的 source 可能具有不同的 strided offset，尤其是 source 已经变成
  // 双缓冲 backing 上动态选择的 view 时。这里重新推导 cloned subview 的 layout，
  // 避免保留依赖旧 source 的类型。
  if (auto subview = dyn_cast<memref::SubViewOp>(cloned)) {
    // subview 的 source 可能已经变成双缓冲 backing 上的 view，因此类型要重算。
    auto sourceType = cast<MemRefType>(subview.getSource().getType());
    subview.getResult().setType(memref::SubViewOp::inferResultType(
        sourceType, subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides()));
  }
  // 将克隆出的地址计算插入到当前 insertion point。
  rewriter.insert(cloned);
  // 建立原结果到新结果的映射，供后续 clone 使用。
  mapping.map(def->getResults(), cloned->getResults());
  // 返回当前 value 在新 IR 中对应的 value。
  return mapping.lookup(value);
}

memref::CopyOp cloneCopyWithMappedSlices(memref::CopyOp copy,
                                         IRRewriter &rewriter,
                                         IRMapping &mapping, Block *sourceBlock,
                                         StringRef role = {}) {
  // 克隆 source 侧所需的 subview/cast/index 等局部定义。
  Value source =
      cloneValueSlice(copy.getSource(), rewriter, mapping, sourceBlock, copy);
  // 克隆 target 侧所需的 subview/cast/index 等局部定义。
  Value target =
      cloneValueSlice(copy.getTarget(), rewriter, mapping, sourceBlock, copy);
  // 在新位置创建等价的 memref.copy。
  auto clonedCopy =
      memref::CopyOp::create(rewriter, copy.getLoc(), source, target);
  // 保留 global_to_shared/shared_to_global 方向，S2 依赖这个属性生成 DMA。
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    clonedCopy->setAttr(kCopyDirectionAttrName, direction);
  // S1 显式记录 copy 的调度角色，避免 S2 再根据操作位置猜测。
  // role 通常是 "prefetch" 或 "store"。
  if (!role.empty())
    clonedCopy->setAttr("db_copy_role", rewriter.getStringAttr(role));
  // 返回新 copy，调用方一般不再需要原 copy。
  return clonedCopy;
}

Operation *cloneOpWithMappedSlices(Operation *op, IRRewriter &rewriter,
                                   IRMapping &mapping, Block *sourceBlock,
                                   bool markCompute = true) {
  // compute op 的 operands 可能依赖局部 subview/cast，需要先克隆这些地址切片。
  for (Value operand : op->getOperands())
    cloneValueSlice(operand, rewriter, mapping, sourceBlock, op);

  // 使用 mapping 克隆 compute op 本身。
  Operation *cloned = op->clone(mapping);
  // 插入到新的双缓冲 loop body 中。
  rewriter.insert(cloned);
  // 将 compute 产生的结果继续映射给后续被克隆操作使用。
  mapping.map(op->getResults(), cloned->getResults());
  // 标记由 S1 克隆出的计算节点，供 S2 恢复调度结构。
  if (markCompute)
    cloned->setAttr("db_compute", UnitAttr::get(op->getContext()));
  // 返回克隆后的 compute op。
  return cloned;
}

/// 单个操作对 tile buffer 的访问类型，读写可以按位组合。
enum TileAccessKind : uint8_t {
  NoTileAccess = 0,
  TileRead = 1,
  TileWrite = 2,
};

enum class ScheduleKind {
  Pointwise,
  Reduction,
};

struct TileAccessNode {
  // 原始 loop body 顶层操作。
  Operation *op;
  // accesses[i] 表示当前操作对第 i 个 tile buffer 的读写方式。
  SmallVector<uint8_t> accesses;
  // 按循环体顶层操作顺序记录 RAW/WAR/WAW 前驱。
  SmallVector<unsigned> predecessors;
  // 是否是输入预取：global_to_shared copy。
  bool isPreload = false;
  // 是否是输入预取 copy 前的 tile setup，例如边界 tile 的 zero fill。
  bool isPreloadSetup = false;
  // 是否是由 preload 数据流驱动的计算。
  bool isCompute = false;
  // 是否是结果回写：shared_to_global copy。
  bool isStore = false;
};

/// 为一个候选循环中的 tile alloc 构建局部 MemorySSA 风格的依赖图。
///
/// 每个 tile buffer 拥有独立的内存版本链。当前只处理循环体顶层的线性
/// 调度，不表达任意 CFG 汇合；遇到无法分类的 tile 访问时保守放弃转换。
struct TileMemorySSAAnalysis {
  TileMemorySSAAnalysis(AliasAnalysis &aliasAnalysis, scf::ForOp forOp,
                        SingleBufferSchedule &schedule)
      // 保存分析所需上下文，并把识别结果写回 schedule。
      : aliasAnalysis(aliasAnalysis), forOp(forOp), schedule(schedule) {}

  // 用于判断 subview/cast 等派生 memref 是否别名到某个 tile。
  AliasAnalysis &aliasAnalysis;
  // 当前正在尝试识别的 loop。
  scf::ForOp forOp;
  // 成功识别后填充的调度信息。
  SingleBufferSchedule &schedule;
  // 从 global_to_shared copy target 中发现的 tile root。
  SmallVector<Value> tileAllocs;
  // tile root value -> tileAllocs 下标。
  llvm::SmallDenseMap<Value, unsigned> tileIndices;
  // 被确认是 preload 的 copy operation 集合。
  llvm::SmallDenseSet<Operation *> preloadCopies;
  // 按原 loop body 顺序保存的 tile 访问节点图。
  SmallVector<TileAccessNode> graph;

  static StringRef getNodeRole(const TileAccessNode &node) {
    // preload 在诊断图中显示为 LOAD。
    if (node.isPreload)
      return "LOAD";
    // preload setup 在诊断图中显示为 LOAD_SETUP。
    if (node.isPreloadSetup)
      return "LOAD_SETUP";
    // compute 在诊断图中显示为 COMPUTE。
    if (node.isCompute)
      return "COMPUTE";
    // store 在诊断图中显示为 STORE。
    if (node.isStore)
      return "STORE";
    // 走到这里通常表示候选 loop 不能被当前 pass 接受。
    return "UNKNOWN";
  }

  static StringRef getAccessName(uint8_t access) {
    // 同一个 op 可能既读又写同一个 tile。
    if (access == (TileRead | TileWrite))
      return "READ_WRITE";
    // 只读 tile。
    if (access == TileRead)
      return "READ";
    // 只写 tile。
    if (access == TileWrite)
      return "WRITE";
    // 不访问 tile。
    return "NONE";
  }

  /// 打印分析得到的节点分类、tile 访问方式和内存依赖流向。
  void print(raw_ostream &os) const {
    os << "\n=== TileMemorySSAAnalysis ===\n";
    os << "loop: ";
    forOp->print(os, OpPrintingFlags().skipRegions());
    os << "\n";

    os << "tiles:\n";
    for (auto [tileIndex, tile] : llvm::enumerate(tileAllocs)) {
      os << "  tile#" << tileIndex << ": ";
      if (Operation *def = tile.getDefiningOp())
        def->print(os, OpPrintingFlags().skipRegions());
      else
        tile.print(os);
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
    // 纯计算如果不触碰内存，也不影响 tile 内存版本。
    return isMemoryEffectFree(op);
  }

  std::optional<unsigned> getTileIndex(Value value) {
    // 只有 memref value 才可能是 tile buffer 或其 view。
    if (!isa<BaseMemRefType>(value.getType()))
      return std::nullopt;
    // subview/cast 等派生值通过 AliasAnalysis 归并到对应 tile alloc。
    for (auto [index, tile] : llvm::enumerate(tileAllocs)) {
      // 只要 value 可能别名到该 tile alloc，就归类到这个 tile。
      if (aliasAnalysis.alias(value, tile))
        return index;
    }
    // 不属于任何被跟踪 tile。
    return std::nullopt;
  }

  bool accessesAnyTile(ArrayRef<uint8_t> accesses) {
    // 只要任意 tile 的访问位非空，该 op 就需要进入依赖图。
    return llvm::any_of(accesses,
                        [](uint8_t access) { return access != NoTileAccess; });
  }

  bool isOverwriteFillSetupOp(Operation *op) {
    if (isa<linalg::FillOp>(op))
      return true;

    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || !ifOp.getElseRegion().empty())
      return false;

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
    return foundFill && onlyFillLikeOps;
  }

  std::optional<unsigned> getSingleSetupTile(const TileAccessNode &node) {
    std::optional<unsigned> setupTile;
    for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
      if (access == NoTileAccess)
        continue;
      bool isOverwrite = access == TileWrite ||
                         (access == (TileRead | TileWrite) &&
                          isOverwriteFillSetupOp(node.op));
      if (!isOverwrite)
        return std::nullopt;
      if (setupTile)
        return std::nullopt;
      setupTile = tileIndex;
    }
    return setupTile;
  }

  bool hasFollowingPreloadForTile(unsigned nodeIndex, unsigned tileIndex) {
    for (unsigned nextIndex = nodeIndex + 1; nextIndex < graph.size();
         ++nextIndex) {
      const TileAccessNode &next = graph[nextIndex];
      if (!next.isPreload)
        continue;
      if (!(next.accesses[tileIndex] & TileWrite))
        continue;
      if (llvm::is_contained(next.predecessors, nodeIndex))
        return true;
    }
    return false;
  }

  bool isPreloadSetupCandidate(unsigned nodeIndex) {
    TileAccessNode &node = graph[nodeIndex];
    if (node.isPreload || node.isStore || isa<memref::CopyOp>(node.op))
      return false;

    auto setupTile = getSingleSetupTile(node);
    if (!setupTile)
      return false;
    return hasFollowingPreloadForTile(nodeIndex, *setupTile);
  }

  void addCopyAccesses(memref::CopyOp copy, TileAccessNode &node) {
    // copy 的 source 是读，target 是写。
    if (auto source = getTileIndex(copy.getSource()))
      node.accesses[*source] |= TileRead;
    if (auto target = getTileIndex(copy.getTarget()))
      node.accesses[*target] |= TileWrite;
  }

  bool addPreciseEffectAccesses(Operation *op, TileAccessNode &node) {
    // 尽量使用 MLIR operation 自己声明的精确 memory effects。
    auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffectOp)
      return false;

    // 收集所有 effect instance。
    SmallVector<MemoryEffects::EffectInstance> effects;
    memoryEffectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      // 这里只关心 Read/Write，忽略 Allocate/Free 等其它 effect。
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
      if (!isRead && !isWrite)
        continue;

      // 没有关联具体 Value 的读写 effect 无法精确归属到某个 tile。
      Value effectValue = effect.getValue();
      if (!effectValue)
        return false;

      // 把 effect value 通过 alias analysis 映射到所有可能的 tile。
      for (auto [index, tile] : llvm::enumerate(tileAllocs)) {
        if (!aliasAnalysis.alias(effectValue, tile))
          continue;
        // 记录 tile read。
        if (isRead)
          node.accesses[index] |= TileRead;
        // 记录 tile write。
        if (isWrite)
          node.accesses[index] |= TileWrite;
      }
    }
    // 所有读写 effect 都能精确归属，返回 true。
    return true;
  }

  void addConservativeModRefAccesses(Operation *op, TileAccessNode &node) {
    // 无法获得带 Value 的精确 effect 时，回退到保守 ModRef 查询。
    for (auto [index, tile] : llvm::enumerate(tileAllocs)) {
      // 对每个 tile 单独询问 op 是否可能读/写它。
      ModRefResult modRef = aliasAnalysis.getModRef(op, tile);
      // Ref 表示可能读取。
      if (modRef.isRef())
        node.accesses[index] |= TileRead;
      // Mod 表示可能写入。
      if (modRef.isMod())
        node.accesses[index] |= TileWrite;
    }
  }

  bool collectCopyIns() {
    // 只在当前 loop body 的顶层线性操作中寻找 preload。
    Block *forBody = forOp.getBody();
    for (Operation &op : forBody->without_terminator()) {
      // 尝试把当前操作当作 memref.copy。
      auto copy = dyn_cast<memref::CopyOp>(&op);
      // 当前双缓冲模式只接受 global -> shared 的输入预取。
      if (!copy || !hasCopyDirection(copy, kGlobalToShared))
        continue;
      // 如果 source 也来自某个 tile alloc，这不是 global 输入预取。
      auto sourceAlloc = findTileRoot(copy.getSource());
      // tile -> tile 或 tile -> external 的 copy 不是输入预取。
      if (sourceAlloc && tileIndices.contains(sourceAlloc))
        continue;
      // target 必须能追溯到一个 tile alloc。
      auto alloc = findTileRoot(copy.getTarget());
      // alloc 必须在 copy 处可用，否则无法安全映射到新双缓冲结构。
      if (!alloc || !isAvailableBeforeCopy(alloc, forOp, copy))
        continue;
      // 同一 tile 出现多个预取定义时，当前线性版本模型无法唯一分类。
      if (tileIndices.contains(alloc))
        return false;
      // 保存 tile alloc 与对应 preload copy。
      schedule.triplets.push_back({alloc, {}, copy});
      // 建立 alloc 到 tile index 的映射。
      tileIndices[alloc] = tileAllocs.size();
      // 记录 tile alloc 列表。
      tileAllocs.push_back(alloc);
      // 记录这个 copy 在访问图中应被分类为 preload。
      preloadCopies.insert(copy.getOperation());
    }

    // 至少有一个输入预取才可能进行双缓冲。
    return !schedule.triplets.empty();
  }

  bool buildAccessGraph() {
    // 重新构图前清空旧结果。
    graph.clear();
    // lastWrite 记录每个 tile 当前的版本定义者；
    // readsSinceWrite 用于在下一次写入时补充 WAR 依赖。
    SmallVector<std::optional<unsigned>> lastWrite(tileAllocs.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tileAllocs.size());

    // 按原始 loop body 顶层顺序扫描，构建线性 MemorySSA 风格依赖。
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // 初始化当前操作的 tile 访问节点。
      TileAccessNode node{
          &op, SmallVector<uint8_t>(tileAllocs.size(), NoTileAccess)};
      // copy 的读写可以直接由 source/target 推出。
      if (auto copy = dyn_cast<memref::CopyOp>(&op))
        addCopyAccesses(copy, node);
      // 其它非 utility op 先用精确 effect，失败再用保守 ModRef。
      else if (!isAllowedLocalUtilityOp(&op) &&
               !addPreciseEffectAccesses(&op, node))
        addConservativeModRefAccesses(&op, node);

      // 不访问任何 tile 的操作不进入图。
      if (!accessesAnyTile(node.accesses))
        continue;

      // 标记输入预取节点。
      node.isPreload = preloadCopies.contains(&op);
      // copy 还可能是结果回写 store。
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        // store 应当从 tile 读。
        bool readsTile = getTileIndex(copy.getSource()).has_value();
        // store 不应当写另一个 tile。
        bool writesTile = getTileIndex(copy.getTarget()).has_value();
        // 结果回写必须是 shared -> global，并且只读取当前 tile。
        node.isStore =
            hasCopyDirection(copy, kSharedToGlobal) && readsTile && !writesTile;
      }

      // 当前节点将被追加到 graph 末尾，因此下标就是当前 graph.size()。
      unsigned nodeIndex = graph.size();
      // 用 set 去重来自多个 tile 的同一前驱。
      llvm::SmallDenseSet<unsigned> predecessors;
      // 对每个 tile 独立维护 RAW/WAR/WAW 依赖。
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        // 当前节点是否读该 tile。
        bool reads = access & TileRead;
        // 当前节点是否写该 tile。
        bool writes = access & TileWrite;
        // 读依赖最近一次写：RAW。
        if (reads && lastWrite[tileIndex])
          predecessors.insert(*lastWrite[tileIndex]);
        if (writes) {
          // 写依赖最近一次写和该版本之后的所有读：WAW + WAR。
          if (lastWrite[tileIndex])
            predecessors.insert(*lastWrite[tileIndex]);
          // 当前写必须排在上一版本之后的所有读之后，避免 WAR 破坏。
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          // 写产生了新版本，旧 readsSinceWrite 不再需要。
          readsSinceWrite[tileIndex].clear();
          // 当前节点成为该 tile 最新写。
          lastWrite[tileIndex] = nodeIndex;
        } else if (reads)
          // 纯读先记录下来，未来遇到写时形成 WAR 依赖。
          readsSinceWrite[tileIndex].push_back(nodeIndex);
      }
      // 固化并排序前驱，保证诊断输出稳定。
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors);
      // 将节点加入图。
      graph.push_back(std::move(node));
    }
    // 图构建完成，具体的 pointwise/reduction 分类由独立 classifier 完成。
    return true;
  }

  bool writesLoopCarriedExternalBuffer(Operation *op) {
    // reduction compute 必须能声明写内存。
    auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memoryEffectOp)
      return false;

    // 查询操作的写 effect。
    SmallVector<MemoryEffects::EffectInstance> effects;
    memoryEffectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      // 只关心写。
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      // 获取被写入的 memref。
      Value value = effect.getValue();
      // 忽略无 value、非 memref、或写入 tracked tile 的 effect。
      if (!value || !isa<BaseMemRefType>(value.getType()) ||
          getTileIndex(value))
        continue;

      // 如果被写 memref 不是 loop body 内定义的，认为它是循环携带的外部 accumulator。
      Operation *def = value.getDefiningOp();
      if (!def || def->getBlock() != forOp.getBody())
        return true;
    }
    // 没有找到外部写。
    return false;
  }

  void classifyPreloadSetupNodes() {
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      if (isPreloadSetupCandidate(nodeIndex))
        node.isPreloadSetup = true;
    }
  }

  llvm::SmallBitVector computeReachableFromPreload() {
    // 从所有 preload 正向传播，找出由输入预取可达的节点。
    // reachableFromPreload[i] 表示节点 i 依赖某个 LOAD。
    llvm::SmallBitVector reachableFromPreload(graph.size());
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      // LOAD 自身天然可达。
      bool reachable = node.isPreload;
      // 如果任一前驱可达，则当前节点也可达。
      for (unsigned predecessor : node.predecessors)
        reachable |= reachableFromPreload.test(predecessor);
      // 记录可达性。
      if (reachable)
        reachableFromPreload.set(nodeIndex);
    }
    return reachableFromPreload;
  }

  llvm::SmallBitVector computeReachesStore() {
    // 从所有 store 沿前驱边反向传播，找出能够影响结果回写的节点。
    // reachesStore[i] 表示节点 i 的结果最终流向某个 STORE。
    llvm::SmallBitVector reachesStore(graph.size());
    // STORE 节点本身作为反向传播起点。
    for (auto [nodeIndex, node] : llvm::enumerate(graph))
      if (node.isStore)
        reachesStore.set(nodeIndex);
    // 反向遍历拓扑序，把 reachesStore 传播到前驱。
    for (int64_t nodeIndex = static_cast<int64_t>(graph.size()) - 1;
         nodeIndex >= 0; --nodeIndex) {
      // 当前节点不流向 STORE 时，不需要传播。
      if (!reachesStore.test(nodeIndex))
        continue;
      // 当前节点流向 STORE，则它的所有前驱也流向 STORE。
      for (unsigned predecessor : graph[nodeIndex].predecessors)
        reachesStore.set(predecessor);
    }
    return reachesStore;
  }

  std::optional<ScheduleKind> inferScheduleKind() {
    bool hasStore = llvm::any_of(
        graph, [](const TileAccessNode &node) { return node.isStore; });
    if (hasStore)
      return ScheduleKind::Pointwise;

    llvm::SmallBitVector reachableFromPreload = computeReachableFromPreload();
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      if (node.isPreload || node.isPreloadSetup)
        continue;
      if (reachableFromPreload.test(nodeIndex) &&
          writesLoopCarriedExternalBuffer(node.op))
        return ScheduleKind::Reduction;
    }
    return std::nullopt;
  }

  bool classifyPointwiseSchedule() {
    llvm::SmallBitVector reachableFromPreload = computeReachableFromPreload();
    llvm::SmallBitVector reachesStore = computeReachesStore();

    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      // LOAD/SETUP/STORE 已经分类，跳过。
      if (node.isPreload || node.isPreloadSetup || node.isStore)
        continue;

      // 普通 pointwise 要求计算既由 preload 驱动，又最终流向 store。
      node.isCompute =
          reachableFromPreload.test(nodeIndex) && reachesStore.test(nodeIndex);

      // copy 只能作为已识别的 preload/store，不能夹在计算数据流内部。
      // 任意无法分类的 tile 访问都会让当前 loop 失去候选资格。
      if (isa<memref::CopyOp>(node.op) || !node.isCompute)
        return false;
    }
    schedule.isReduction = false;
    return true;
  }

  bool classifyReductionSchedule() {
    llvm::SmallBitVector reachableFromPreload = computeReachableFromPreload();
    bool hasReductionCompute = false;
    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      // LOAD/SETUP 已经分类，跳过。
      if (node.isPreload || node.isPreloadSetup)
        continue;

      // Reduction 流没有循环内 store，而是原地更新循环外 accumulator。
      node.isCompute = reachableFromPreload.test(nodeIndex) &&
                       writesLoopCarriedExternalBuffer(node.op);
      hasReductionCompute |= node.isCompute;

      // copy 只能作为已识别的 preload，不能夹在计算数据流内部。
      // 任意无法分类的 tile 访问都会让当前 loop 失去候选资格。
      if (isa<memref::CopyOp>(node.op) || !node.isCompute)
        return false;
    }
    schedule.isReduction = true;
    return hasReductionCompute;
  }

  bool classifySchedule() {
    classifyPreloadSetupNodes();
    std::optional<ScheduleKind> kind = inferScheduleKind();
    if (!kind)
      return false;
    return *kind == ScheduleKind::Reduction ? classifyReductionSchedule()
                                            : classifyPointwiseSchedule();
  }

  bool refineCopyInsToComputeUsedBuffers() {
    // 删除没有被计算节点实际访问的预取，避免为无关 copy 分配双缓冲。
    // usedByCompute[i] 表示第 i 个 tile 被 compute 节点访问。
    llvm::SmallBitVector usedByCompute(tileAllocs.size());
    for (const TileAccessNode &node : graph) {
      // 只看 compute 节点，不因 preload/store 误保留 tile。
      if (!node.isCompute)
        continue;
      // compute 读或写某 tile，都说明该 tile 是双缓冲候选。
      for (auto [index, access] : llvm::enumerate(node.accesses))
        if (access != NoTileAccess)
          usedByCompute.set(index);
    }

    // 准备压缩后的 schedule 和 tile bookkeeping。
    SmallVector<ScheduleTriplet> refinedTriplets;
    SmallVector<Value> refinedAllocs;
    llvm::SmallDenseMap<Value, unsigned> refinedIndices;
    llvm::SmallDenseSet<Operation *> refinedPreloadCopies;
    for (auto [index, triplet] : llvm::enumerate(schedule.triplets)) {
      // 跳过 compute 完全没用到的 preload。
      if (!usedByCompute.test(index))
        continue;
      // 为保留下来的 tile 分配新的连续下标。
      refinedIndices[triplet.tile] = refinedAllocs.size();
      // 保留 alloc/copy 调度对。
      refinedTriplets.push_back(triplet);
      // 保留 tile alloc。
      refinedAllocs.push_back(triplet.tile);
      // 保留 preload copy 标记。
      refinedPreloadCopies.insert(triplet.load.getOperation());
    }
    // 如果没有任何 tile 被 compute 使用，则不是有效候选。
    if (refinedTriplets.empty())
      return false;
    // 用精简后的结果覆盖旧 schedule。
    schedule.triplets = std::move(refinedTriplets);
    tileAllocs = std::move(refinedAllocs);
    tileIndices = std::move(refinedIndices);
    preloadCopies = std::move(refinedPreloadCopies);
    // 精简成功。
    return true;
  }

  bool collectScheduleFromGraph() {
    // DAG 已完成分类，此处按原始拓扑顺序生成改写 schedule。
    // 先清掉可能来自前一次尝试的结果。
    schedule.computeOps.clear();
    schedule.stores.clear();
    for (const TileAccessNode &node : graph) {
      // preload setup 会随对应的 load 一起克隆到 prologue/next-prefetch。
      if (node.isPreloadSetup) {
        auto tile = getSingleSetupTile(node);
        if (!tile)
          return false;
        schedule.triplets[*tile].setupOps.push_back(node.op);
      }
      // compute op 会被克隆到新 loop 的 current-buffer 阶段。
      if (node.isCompute)
        schedule.computeOps.push_back(node.op);
      // store copy 会被克隆到 compute 之后；reduction 模式下可以没有。
      if (node.isStore)
        schedule.stores.push_back(cast<memref::CopyOp>(node.op));
    }
    // 必须有 compute；普通 generic 还必须有 store，reduction 可以无 store。
    return !schedule.computeOps.empty() &&
           (schedule.isReduction || !schedule.stores.empty());
  }

  bool validateScheduleDependencies() {
    // 每个 tile 必须满足 preload -> compute -> store 的阶段顺序。
    // seenPreload/Compute/Store 分别记录每个 tile 已经出现过的阶段。
    llvm::SmallBitVector seenPreload(tileAllocs.size());
    llvm::SmallBitVector seenCompute(tileAllocs.size());
    llvm::SmallBitVector seenStore(tileAllocs.size());

    for (auto [nodeIndex, node] : llvm::enumerate(graph)) {
      // 局部图应天然保持拓扑序；出现反向边表示构图或分类不一致。
      // 任一前驱下标大于等于当前节点，说明依赖顺序不合法。
      if (llvm::any_of(node.predecessors, [&](unsigned predecessor) {
            return predecessor >= nodeIndex;
          }))
        return false;

      // 按 tile 检查阶段顺序。
      for (auto [index, access] : llvm::enumerate(node.accesses)) {
        // 当前节点不访问该 tile，跳过。
        if (access == NoTileAccess)
          continue;
        if (node.isPreload) {
          // preload 必须写 tile，且不能发生在该 tile 的 compute/store 之后。
          if (!(access & TileWrite) || seenPreload.test(index) ||
              seenCompute.test(index) || seenStore.test(index))
            return false;
          // 标记该 tile 已完成 preload。
          seenPreload.set(index);
          continue;
        }
        if (node.isPreloadSetup) {
          // setup 必须是 preload 前的纯 tile 写，不能读旧 tile 内容。
          if (!(access & TileWrite) || seenPreload.test(index) ||
              seenCompute.test(index) || seenStore.test(index))
            return false;
          continue;
        }
        if (node.isCompute) {
          // compute 必须发生在 preload 之后、store 之前。
          if (!seenPreload.test(index) || seenStore.test(index))
            return false;
          // 标记该 tile 已参与 compute。
          seenCompute.set(index);
          continue;
        }
        if (node.isStore) {
          // store 必须读取 tile，且该 tile 必须已经被 compute。
          if (!(access & TileRead) || !seenCompute.test(index))
            return false;
          // 标记该 tile 已 store。
          seenStore.set(index);
        }
      }
    }

    // 所有参与双缓冲的 tile 都必须有 preload。
    if (seenPreload.count() != tileAllocs.size())
      return false;
    // reduction 形态要求有 compute 且没有 loop 内 store。
    if (schedule.isReduction)
      return seenCompute.count() != 0 && seenStore.none();
    // 只要求被回写的 tile 确实经过计算；纯输入 tile 不要求 store。
    for (memref::CopyOp store : schedule.stores) {
      // store 的 source 必须是某个被跟踪 tile。
      auto tile = getTileIndex(store.getSource());
      // store 对应 tile 必须经过 compute。
      if (!tile || !seenCompute.test(*tile))
        return false;
    }
    // 普通 generic 调度验证通过。
    return true;
  }

  bool run() {
    // 第一次构图用于剔除未参与计算的预取；收缩 tile 集合后重新构图，
    // 保证最终节点索引和访问向量一致。
    // 任一阶段失败，都表示当前 loop 不适合这个 pass 改写。
    if (!collectCopyIns() || !buildAccessGraph() || !classifySchedule() ||
        !refineCopyInsToComputeUsedBuffers() || !buildAccessGraph() ||
        !classifySchedule() || !collectScheduleFromGraph() ||
        !validateScheduleDependencies())
      return false;

    // 打印识别出的 tile MemorySSA 图，方便观察 LOAD/COMPUTE/STORE 关系。
    print(llvm::outs());
    // 告诉调用方这个 loop 可以改写。
    return true;
  }
};

/// 将识别出的单缓冲 loop 改写为双缓冲 loop。
void rewriteAsDoubleBuffered(IRRewriter &rewriter, scf::ForOp sbForOp,
                             SingleBufferSchedule &schedule, int &uid) {
  // 复用原 loop 的位置信息，便于诊断映射。
  auto loc = sbForOp.getLoc();
  // 取 context 用于创建 UnitAttr 等属性。
  auto context = sbForOp.getContext();

  // cur selector 用 i1 表示，true/false 分别选择两个物理 slot。
  auto boolType = rewriter.getI1Type();

  // 保护调用方 insertion point，函数结束后自动恢复。
  RewriterBase::InsertionGuard guard(rewriter);
  // 新的 prologue、backing alloc、kernel loop 都插在原 loop 之前。
  rewriter.setInsertionPoint(sbForOp);

  // 定义后续重写通用的常量。
  // 初始 cur=true；约定 true 时 slot0 是 current，slot1 是 next。
  Value trueVal = arith::ConstantOp::create(rewriter, loc, boolType,
                                            rewriter.getBoolAttr(true));
  // 每个 tile 分配一个 backing memref；backing 第一维连续存放两个物理 slot。
  // 这里固定使用 2048 字节对齐，匹配后续 Hexagon/VTCM 访问要求。
  int64_t alignment = 2048;
  // memref.alloc 的 alignment 属性。
  auto alignmentAttr = rewriter.getI64IntegerAttr(alignment);
  // 第一轮预取使用 slot0，所以 leading offset 为 0。
  Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
  // 每个 tile 对应一个两倍大小的 backing allocation。
  SmallVector<Value, 3> backingBuffers;
  // prologue 中用于第 0 次迭代预取的 slot0 view。
  SmallVector<Value, 3> firstBuffers;
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    // 取出原始 tile alloc。
    Value tile = schedule.triplets[i].tile;
    // 原始 tile 的 memref 类型。
    auto tileType = cast<MemRefType>(tile.getType());
    // backing shape 初始等于 tile shape。
    SmallVector<int64_t> backingShape(tileType.getShape());
    // 第一维扩大两倍，用一块 allocation 表示 ping/pong 两个 slot。
    backingShape.front() *= 2;
    // backing 保持原 element type、layout 和 memory space。
    auto backingType =
        MemRefType::get(backingShape, tileType.getElementType(),
                        tileType.getLayout(), tileType.getMemorySpace());
    // 创建新的双缓冲 backing allocation。
    Value backing = memref::AllocOp::create(
        rewriter, loc, backingType, mlir::ValueRange{}, alignmentAttr);
    // 记录 backing，kernel 中会根据 cur 从它切 current/next view。
    backingBuffers.push_back(backing);
    // 切出 slot0 view，供 prologue 预取第一轮 tile。
    firstBuffers.push_back(
        createDoubleBufferView(rewriter, loc, backing, tileType, zeroIndex));
  }

  // 从原始单缓冲 for loop 读取 bounds。
  // 新 kernel loop 复用原 loop 的上下界和步长。
  Value lowerBound = sbForOp.getLowerBound();
  Value upperBound = sbForOp.getUpperBound();
  Value step = sbForOp.getStep();
  // 原 loop induction variable，克隆原始操作时会映射到 lowerBound/dbIndVar/nextIdx。
  auto indVar = sbForOp.getInductionVar();
  // 给这组 prologue/kernel 打同一个 id，S2 可用来恢复调度关系。
  auto idAttr = mlir::IntegerAttr::get(rewriter.getI64Type(), uid++);

  // Prologue：仅当原 loop 不是 0 次迭代时执行。
  // 如果 lowerBound < upperBound，说明 loop 至少执行一次。
  Value mayLoop =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            lowerBound, upperBound)
          .getResult();
  // prologue 只在非空 loop 时预取第一个 tile。
  auto ifMayLoop =
      scf::IfOp::create(rewriter, loc, TypeRange(), mayLoop, false);
  // 标记这是同一个 double-buffer generic 调度的一部分。
  ifMayLoop->setAttr("db_generic", idAttr);
  // 标记该 if 是 prologue，S2 会按这个结构找第一轮 DMA。
  ifMayLoop->setAttr("db_prologue", UnitAttr::get(context));
  // 在 prologue then block 中插入第一轮 preload。
  rewriter.setInsertionPointToStart(&ifMayLoop.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    // 每个 preload 独立 mapping，避免不同 tile 的局部地址计算互相污染。
    IRMapping mapping;
    // 第一轮的原 induction variable 等价于 lowerBound。
    mapping.map(indVar, lowerBound);
    // 原 tile alloc 映射到 slot0 view。
    mapping.map(schedule.triplets[i].tile, firstBuffers[i]);
    // 克隆第一轮 copy-in 前的 tile setup。
    for (Operation *setup : schedule.triplets[i].setupOps)
      cloneOpWithMappedSlices(setup, rewriter, mapping, sbForOp.getBody(),
                              /*markCompute=*/false);
    // 克隆原 load copy，并标记为 prefetch。
    cloneCopyWithMappedSlices(schedule.triplets[i].load, rewriter, mapping,
                              sbForOp.getBody(), "prefetch");
  }

  // Kernel：通过 loop-carried selector 选择 current buffer，而不是分别克隆
  // ping/pong 两份 kernel。这样 compute slice 只保留一份克隆。
  // 在原 loop 位置创建新的 kernel loop。
  rewriter.setInsertionPoint(sbForOp);
  // 新 loop 多携带一个 i1 iter_arg：当前 buffer selector。
  auto dbLoop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step,
                                   ValueRange{trueVal});
  // 与 prologue 使用同一个调度 id。
  dbLoop->setAttr("db_generic", idAttr);
  // 取得新 loop body。
  Block *forBody = dbLoop.getBody();
  // scf::ForOp::create 会生成默认 yield，这里删除后重建 body。
  if (!forBody->empty() && isa<scf::YieldOp>(forBody->back()))
    forBody->back().erase();
  // 开始填充新 loop body。
  rewriter.setInsertionPointToStart(forBody);
  // 新 loop 的 induction variable。
  Value dbIndVar = dbLoop.getInductionVar();
  // loop-carried selector，表示哪一个 slot 是 next/current。
  Value cur = dbLoop.getRegionIterArgs().front();

  // cur=true 表示 slot0 是 current、slot1 是 next。这里把 selector 转成 offset，
  // 而不是在两个 memref SSA value 之间做选择。
  // nextSlot = cur；true cast 成 1，false cast 成 0。
  Value nextSlot =
      arith::IndexCastUIOp::create(rewriter, loc, rewriter.getIndexType(), cur);
  // currentSlotBit = !cur；与 nextSlot 相反。
  Value currentSlotBit =
      arith::XOrIOp::create(rewriter, loc, cur, trueVal);
  // currentSlot 转成 index 后可用于 offset 计算。
  Value currentSlot = arith::IndexCastUIOp::create(
      rewriter, loc, rewriter.getIndexType(), currentSlotBit);

  // currentBuffers 供本轮 compute/store 使用。
  SmallVector<Value, 3> currentBuffers;
  // nextBuffers 供下一轮 prefetch 使用。
  SmallVector<Value, 3> nextBuffers;
  for (auto [backing, triplet] :
       llvm::zip(backingBuffers, schedule.triplets)) {
    // 取原 tile 类型以得到 slot 大小。
    auto tileType = cast<MemRefType>(triplet.tile.getType());
    // 一个 slot 在 backing 第一维上占 tileType.getDimSize(0)。
    Value tileExtent =
        arith::ConstantIndexOp::create(rewriter, loc, tileType.getDimSize(0));
    // current slot 的第一维 offset。
    Value currentOffset =
        arith::MulIOp::create(rewriter, loc, currentSlot, tileExtent);
    // next slot 的第一维 offset。
    Value nextOffset =
        arith::MulIOp::create(rewriter, loc, nextSlot, tileExtent);
    // 从 backing 中切出当前轮 tile view。
    currentBuffers.push_back(createDoubleBufferView(
        rewriter, loc, backing, tileType, currentOffset));
    // 从 backing 中切出下一轮 tile view。
    nextBuffers.push_back(createDoubleBufferView(rewriter, loc, backing,
                                                 tileType, nextOffset));
  }

  // 判断是否需要执行下一轮 preload；最后一轮没有下一轮 preload。
  // 下一轮原始迭代下标。
  Value nextIdx =
      arith::AddIOp::create(rewriter, loc, dbIndVar, step).getResult();
  // 只有 nextIdx < upperBound 时才存在下一轮需要预取。
  Value nextExists =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            nextIdx, upperBound)
          .getResult();

  // ifNext 包住下一轮 prefetch，最后一轮不执行。
  auto ifNext =
      scf::IfOp::create(rewriter, loc, TypeRange(), nextExists, false);
  // 标记这是 kernel 内的 prefetch 区域。
  ifNext->setAttr("db_prefetch", UnitAttr::get(context));
  // 在 ifNext then block 内克隆下一轮 load。
  rewriter.setInsertionPointToStart(&ifNext.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    // 每个 tile 的下一轮 preload 独立克隆。
    IRMapping mapping;
    // 原 induction variable 映射到 nextIdx。
    mapping.map(indVar, nextIdx);
    // 原 tile alloc 映射到 next slot view。
    mapping.map(schedule.triplets[i].tile, nextBuffers[i]);
    // 克隆下一轮 copy-in 前的 tile setup。
    for (Operation *setup : schedule.triplets[i].setupOps)
      cloneOpWithMappedSlices(setup, rewriter, mapping, sbForOp.getBody(),
                              /*markCompute=*/false);
    // 克隆 load copy，仍标记为 prefetch。
    cloneCopyWithMappedSlices(schedule.triplets[i].load, rewriter, mapping,
                              sbForOp.getBody(), "prefetch");
  }

  // 在下一轮 tile prefetch 启动后，计算并写回当前 tile。S2 会在这个边界插入
  // 当前 DMA 的 wait。
  // prefetch if 之后插入当前轮 compute/store。
  rewriter.setInsertionPointAfter(ifNext);
  // 当前轮 compute/store 共用同一个 mapping，保证它们之间的 SSA 结果能串起来。
  IRMapping mapping;
  // 原 induction variable 映射到新 loop 的 dbIndVar。
  mapping.map(indVar, dbIndVar);
  // 所有原 tile alloc 映射到 current slot view。
  for (auto i = 0; i < schedule.triplets.size(); ++i)
    mapping.map(schedule.triplets[i].tile, currentBuffers[i]);
  // 克隆 compute 操作；普通 generic 和 reduction 都会走这里。
  for (Operation *compute : schedule.computeOps)
    cloneOpWithMappedSlices(compute, rewriter, mapping, sbForOp.getBody());
  // 普通 generic 克隆 store；reduction 的 stores 为空，因此不会产生 store。
  for (memref::CopyOp store : schedule.stores)
    cloneCopyWithMappedSlices(store, rewriter, mapping, sbForOp.getBody(),
                              "store");

  // 翻转 selector，下一轮 current/next slot 互换。
  Value nextCur = arith::XOrIOp::create(rewriter, loc, cur, trueVal);
  // 将翻转后的 selector yield 给下一轮。
  scf::YieldOp::create(rewriter, loc, nextCur);
  // 新双缓冲结构已经替代原 loop，删除原单缓冲 loop。
  rewriter.eraseOp(sbForOp);
}

// 解析 tiled_generic 风格 loop 的调度入口。
bool generateSchedule(IRRewriter &rewriter, scf::ForOp forOp,
                      SingleBufferSchedule &schedule,
                      AliasAnalysis &aliasAnalysis) {
  // 当前 rewrite 只会自己新增一个 selector iter_arg，不支持原 loop 已有 iter_args。
  if (!forOp.getInitArgs().empty())
    return false;
  // 保存原始 loop，后续 rewriteAsDoubleBuffered 会用它读取 bounds/body。
  schedule.forOp = forOp;

  // 对当前 loop 进行 tile MemorySSA 分析，并把识别出的调度写入 schedule。
  TileMemorySSAAnalysis analysis{aliasAnalysis, forOp, schedule};
  // 分析失败表示该 loop 不满足当前 pass 支持的双缓冲形态。
  if (!analysis.run())
    return false;

  // 合并后的 backing allocation 目前要求非标量静态 tile shape，这样第二个 slot
  // 才有编译期可确定的 leading offset。
  // backing 第一维需要乘 2，因此 tile 必须是 rank>0 且静态 shape。
  return llvm::all_of(schedule.triplets, [](ScheduleTriplet triplet) {
    // 读取原 tile alloc 的 memref 类型。
    auto type = cast<MemRefType>(triplet.tile.getType());
    // 标量或动态 shape 暂不支持创建稳定的双 slot backing。
    return type.getRank() > 0 && type.hasStaticShape();
  });
}

struct HexagonDoubleBufferGenericS1Pass
    : public ::impl::HexagonDoubleBufferGenericS1Base<
          HexagonDoubleBufferGenericS1Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    int uniqueID = 0;
    auto func = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    func.walk([&uniqueID, &aliasAnalysis](scf::ForOp forOp) {
      SingleBufferSchedule schedule;
      IRRewriter rewriter(forOp.getContext());
      bool viableCandidate =
          generateSchedule(rewriter, forOp, schedule, aliasAnalysis);
      if (viableCandidate)
        rewriteAsDoubleBuffered(rewriter, forOp, schedule, uniqueID);
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS1Pass() {
  return std::make_unique<HexagonDoubleBufferGenericS1Pass>();
}

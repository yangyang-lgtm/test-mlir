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

constexpr StringLiteral kPlanIdAttr = "db_plan_id";
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue";
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch";
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute";
constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role";

enum TileAccessKind : uint8_t {
  NoTileAccess = 0,
  TileRead = 1,
  TileWrite = 2,
};

struct TileNode {
  Operation *op = nullptr;
  SmallVector<uint8_t> accesses;
  SmallVector<unsigned> predecessors;
  bool isLoad = false;
  bool isLoadSetup = false;
  bool isCompute = false;
  bool isStore = false;
};

struct TileInfo {
  Value root;
  memref::CopyOp load;
  SmallVector<Operation *> setupOps;
};

struct Plan {
  scf::ForOp loop;
  SmallVector<TileInfo> tiles;
  SmallVector<TileNode> nodes;
  SmallVector<Operation *> computes;
  SmallVector<memref::CopyOp> stores;
  bool isReduction = false;
};

bool hasCopyDirection(memref::CopyOp copy, StringRef direction) {
  auto attr = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return attr && attr.getValue() == direction;
}

Value findMemoryRoot(Value value) {
  while (true) {
    if (!value)
      return {};
    if (auto alloc = value.getDefiningOp<memref::AllocOp>())
      return alloc.getMemref();
    if (auto view = value.getDefiningOp<memref::ViewOp>())
      return view.getResult();
    Operation *def = value.getDefiningOp();
    if (!def)
      return {};
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      continue;
    }
    return {};
  }
}

Value findAliasRoot(Value value) {
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return value;
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpret.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def))
      return view.getResult();
    return value;
  }
  return {};
}

bool isAvailableBeforeCopy(Value tile, scf::ForOp loop, memref::CopyOp copy) {
  Operation *def = tile.getDefiningOp();
  if (!def)
    return false;
  Block *body = loop.getBody();
  if (def->getBlock() == body)
    return def->isBeforeInBlock(copy);
  return def->getBlock() == loop->getBlock() && def->isBeforeInBlock(loop);
}

StringRef accessName(uint8_t access) {
  if (access == (TileRead | TileWrite))
    return "READ_WRITE";
  if (access == TileRead)
    return "READ";
  if (access == TileWrite)
    return "WRITE";
  return "NONE";
}

struct TileDataFlowAnalysis {
  TileDataFlowAnalysis(scf::ForOp loop, AliasAnalysis &aliasAnalysis)
      : loop(loop), aliasAnalysis(aliasAnalysis) {}

  bool run(Plan &plan) {
    plan.loop = loop;
    if (!collectTiles(plan) || !buildNodes(plan) || !classify(plan) ||
        !validateGlobalLoadStoreIndependence(plan))
      return false;
    print(plan, llvm::outs());
    return true;
  }

  std::optional<unsigned> getTileIndex(Value value, const Plan &plan) {
    value = findMemoryRoot(value);
    if (!value)
      return std::nullopt;
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      if (value == tile.root || aliasAnalysis.alias(value, tile.root))
        return index;
    return std::nullopt;
  }

  bool collectTiles(Plan &plan) {
    for (Operation &op : loop.getBody()->without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy || !hasCopyDirection(copy, kGlobalToShared))
        continue;

      Value root = findMemoryRoot(copy.getTarget());
      if (!root || !isAvailableBeforeCopy(root, loop, copy))
        return false;
      if (llvm::any_of(plan.tiles,
                       [&](TileInfo tile) { return tile.root == root; }))
        return false;
      plan.tiles.push_back(TileInfo{root, copy});
    }
    return !plan.tiles.empty();
  }

  void addCopyAccesses(memref::CopyOp copy, TileNode &node, Plan &plan) {
    if (auto source = getTileIndex(copy.getSource(), plan))
      node.accesses[*source] |= TileRead;
    if (auto target = getTileIndex(copy.getTarget(), plan))
      node.accesses[*target] |= TileWrite;
  }

  bool addEffectAccesses(Operation *op, TileNode &node, Plan &plan) {
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return false;

    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
      bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
      if (!isRead && !isWrite)
        continue;

      Value value = effect.getValue();
      if (!value || !isa<BaseMemRefType>(value.getType()))
        return false;

      if (auto tile = getTileIndex(value, plan)) {
        if (isRead)
          node.accesses[*tile] |= TileRead;
        if (isWrite)
          node.accesses[*tile] |= TileWrite;
      }
    }
    return true;
  }

  void addConservativeAccesses(Operation *op, TileNode &node, Plan &plan) {
    for (auto [index, tile] : llvm::enumerate(plan.tiles)) {
      ModRefResult modRef = aliasAnalysis.getModRef(op, tile.root);
      if (modRef.isRef())
        node.accesses[index] |= TileRead;
      if (modRef.isMod())
        node.accesses[index] |= TileWrite;
    }
  }

  bool buildNodes(Plan &plan) {
    SmallVector<std::optional<unsigned>> lastWrite(plan.tiles.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(plan.tiles.size());

    for (Operation &op : loop.getBody()->without_terminator()) {
      TileNode node;
      node.op = &op;
      node.accesses.assign(plan.tiles.size(), NoTileAccess);

      if (auto copy = dyn_cast<memref::CopyOp>(&op))
        addCopyAccesses(copy, node, plan);
      else if (!isa<memref::AllocOp, memref::DeallocOp, memref::SubViewOp,
                    memref::ViewOp, memref::CastOp,
                    memref::ReinterpretCastOp>(&op) &&
               !isMemoryEffectFree(&op)) {
        if (!addEffectAccesses(&op, node, plan))
          addConservativeAccesses(&op, node, plan);
      } else if (!isMemoryEffectFree(&op) &&
                 !addEffectAccesses(&op, node, plan)) {
        addConservativeAccesses(&op, node, plan);
      }

      if (!llvm::any_of(node.accesses,
                        [](uint8_t access) { return access != NoTileAccess; }))
        continue;

      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        node.isLoad = hasCopyDirection(copy, kGlobalToShared);
        node.isStore = hasCopyDirection(copy, kSharedToGlobal);
      }

      unsigned nodeIndex = plan.nodes.size();
      llvm::SmallDenseSet<unsigned> predecessors;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        bool reads = access & TileRead;
        bool writes = access & TileWrite;
        if (reads && lastWrite[tileIndex])
          predecessors.insert(*lastWrite[tileIndex]);
        if (writes) {
          if (lastWrite[tileIndex])
            predecessors.insert(*lastWrite[tileIndex]);
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
          if (reads)
            readsSinceWrite[tileIndex].push_back(nodeIndex);
        } else if (reads) {
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors);
      plan.nodes.push_back(std::move(node));
    }
    return true;
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

  std::optional<unsigned> getSingleSetupTile(const TileNode &node) {
    std::optional<unsigned> setupTile;
    for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
      if (access == NoTileAccess)
        continue;
      bool overwritesTile = access == TileWrite ||
                            (access == (TileRead | TileWrite) &&
                             isOverwriteFillSetupOp(node.op));
      if (!overwritesTile)
        return std::nullopt;
      if (setupTile)
        return std::nullopt;
      setupTile = tileIndex;
    }
    return setupTile;
  }

  bool hasFollowingLoadForTile(unsigned nodeIndex, unsigned tileIndex,
                               const Plan &plan) {
    for (unsigned nextIndex = nodeIndex + 1; nextIndex < plan.nodes.size();
         ++nextIndex) {
      const TileNode &next = plan.nodes[nextIndex];
      if (!next.isLoad)
        continue;
      if (next.accesses[tileIndex] & TileWrite)
        return true;
    }
    return false;
  }

  bool classify(Plan &plan) {
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes)) {
      if (node.isLoad) {
        if (!isa<memref::CopyOp>(node.op))
          return false;
        continue;
      }
      if (node.isStore) {
        auto copy = cast<memref::CopyOp>(node.op);
        if (!getTileIndex(copy.getSource(), plan) ||
            getTileIndex(copy.getTarget(), plan))
          return false;
        plan.stores.push_back(copy);
        continue;
      }
      if (auto setupTile = getSingleSetupTile(node);
          setupTile && hasFollowingLoadForTile(nodeIndex, *setupTile, plan)) {
        node.isLoadSetup = true;
        plan.tiles[*setupTile].setupOps.push_back(node.op);
        continue;
      }
      if (isa<memref::CopyOp>(node.op))
        return false;
      node.isCompute = true;
      plan.computes.push_back(node.op);
    }
    return !plan.computes.empty() && (plan.isReduction || !plan.stores.empty());
  }

  bool validateGlobalLoadStoreIndependence(const Plan &plan) {
    for (TileInfo tile : plan.tiles) {
      Value loadSource = findAliasRoot(tile.load.getSource());
      if (!loadSource)
        return false;
      for (memref::CopyOp store : plan.stores) {
        Value storeTarget = findAliasRoot(store.getTarget());
        if (!storeTarget || loadSource == storeTarget ||
            aliasAnalysis.alias(loadSource, storeTarget).isMust())
          return false;
      }
    }
    return true;
  }

  void print(const Plan &plan, raw_ostream &os) {
    os << "\n=== HexagonDoubleBufferPlanRewrite dataflow ===\n";
    os << "loop: ";
    loop->print(os, OpPrintingFlags().skipRegions());
    os << "\n";
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

    os << "dataflow:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(plan.nodes)) {
      for (unsigned predecessor : node.predecessors) {
        const TileNode &source = plan.nodes[predecessor];
        for (auto [tileIndex, sourceAccess] : llvm::enumerate(source.accesses)) {
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

  scf::ForOp loop;
  AliasAnalysis &aliasAnalysis;
};

bool isDefinedBefore(Operation *op, Block *block, Operation *limit) {
  return op && op->getBlock() == block && op->isBeforeInBlock(limit);
}

Value cloneSlice(Value value, IRRewriter &rewriter, IRMapping &mapping,
                 Block *sourceBlock, Operation *limit) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  Operation *def = value.getDefiningOp();
  if (!isDefinedBefore(def, sourceBlock, limit))
    return value;

  for (Value operand : def->getOperands())
    cloneSlice(operand, rewriter, mapping, sourceBlock, limit);

  Operation *cloned = def->clone(mapping);
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

memref::CopyOp cloneCopy(memref::CopyOp copy, IRRewriter &rewriter,
                         IRMapping &mapping, Block *sourceBlock,
                         StringRef role) {
  Value source = cloneSlice(copy.getSource(), rewriter, mapping, sourceBlock,
                            copy.getOperation());
  Value target = cloneSlice(copy.getTarget(), rewriter, mapping, sourceBlock,
                            copy.getOperation());
  auto cloned = memref::CopyOp::create(rewriter, copy.getLoc(), source, target);
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    cloned->setAttr(kCopyDirectionAttrName, direction);
  cloned->setAttr(kPlanCopyRoleAttr, rewriter.getStringAttr(role));
  return cloned;
}

Operation *cloneMappedOp(Operation *op, IRRewriter &rewriter,
                         IRMapping &mapping, Block *sourceBlock,
                         bool markCompute) {
  for (Value operand : op->getOperands())
    cloneSlice(operand, rewriter, mapping, sourceBlock, op);
  Operation *cloned = op->clone(mapping);
  rewriter.insert(cloned);
  mapping.map(op->getResults(), cloned->getResults());
  if (markCompute)
    cloned->setAttr(kPlanComputeAttr, UnitAttr::get(op->getContext()));
  return cloned;
}

Value createSlotView(IRRewriter &rewriter, Location loc, Value backing,
                     MemRefType tileType, Value offset) {
  SmallVector<OpFoldResult> offsets(tileType.getRank(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(tileType.getRank(), rewriter.getIndexAttr(1));
  offsets.front() = offset;
  for (int64_t dim : tileType.getShape())
    sizes.push_back(rewriter.getIndexAttr(dim));
  auto backingType = cast<MemRefType>(backing.getType());
  auto viewType = memref::SubViewOp::inferResultType(backingType, offsets,
                                                     sizes, strides);
  return memref::SubViewOp::create(rewriter, loc, viewType, backing, offsets,
                                   sizes, strides);
}

bool rewritePointwisePlan(IRRewriter &rewriter, Plan &plan, int64_t planId) {
  scf::ForOp loop = plan.loop;
  Location loc = loop.getLoc();
  Block *sourceBlock = loop.getBody();
  MLIRContext *context = loop.getContext();

  SmallVector<Value> backings;
  SmallVector<MemRefType> tileTypes;
  rewriter.setInsertionPoint(loop);
  auto alignmentAttr = rewriter.getI64IntegerAttr(2048);
  for (TileInfo tile : plan.tiles) {
    auto tileType = dyn_cast<MemRefType>(tile.root.getType());
    if (!tileType || tileType.getRank() == 0 || !tileType.hasStaticShape())
      return false;
    SmallVector<int64_t> shape(tileType.getShape().begin(),
                               tileType.getShape().end());
    shape.front() *= 2;
    auto backingType =
        MemRefType::get(shape, tileType.getElementType(), tileType.getLayout(),
                        tileType.getMemorySpace());
    backings.push_back(memref::AllocOp::create(
        rewriter, loc, backingType, ValueRange{}, alignmentAttr));
    tileTypes.push_back(tileType);
  }

  auto idAttr = IntegerAttr::get(IntegerType::get(context, 64), planId);
  Value trueValue = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getBoolAttr(true));
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value first = loop.getLowerBound();

  Value hasFirst =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, first,
                            loop.getUpperBound());
  auto prologue = scf::IfOp::create(rewriter, loc, hasFirst,
                                    /*withElseRegion=*/false);
  prologue->setAttr(kPlanIdAttr, idAttr);
  prologue->setAttr(kPlanPrologueAttr, UnitAttr::get(context));
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(prologue.thenBlock());
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), first);
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      mapping.map(tile.root,
                  createSlotView(rewriter, loc, backings[index],
                                 tileTypes[index], zero));
    for (TileInfo tile : plan.tiles) {
      for (Operation *setup : tile.setupOps)
        cloneMappedOp(setup, rewriter, mapping, sourceBlock,
                      /*markCompute=*/false);
      cloneCopy(tile.load, rewriter, mapping, sourceBlock, kPrefetchRole);
    }
  }

  SmallVector<Value> initArgs{trueValue};
  auto dbLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(),
                                   loop.getUpperBound(), loop.getStep(),
                                   initArgs);
  dbLoop->setAttr(kPlanIdAttr, idAttr);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(dbLoop.getBody());
    Value cur = dbLoop.getRegionIterArgs().front();
    Value next = arith::XOrIOp::create(rewriter, loc, cur, trueValue);
    Value nextIndex = arith::IndexCastUIOp::create(rewriter, loc,
                                                   rewriter.getIndexType(), cur);
    Value curIndex = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getIndexType(), next);

    SmallVector<Value> currentViews;
    SmallVector<Value> nextViews;
    for (auto [index, tileType] : llvm::enumerate(tileTypes)) {
      Value tileSize = arith::ConstantIndexOp::create(
          rewriter, loc, tileType.getShape().front());
      Value currentOffset =
          arith::MulIOp::create(rewriter, loc, curIndex, tileSize);
      Value nextOffset =
          arith::MulIOp::create(rewriter, loc, nextIndex, tileSize);
      currentViews.push_back(
          createSlotView(rewriter, loc, backings[index], tileType,
                         currentOffset));
      nextViews.push_back(
          createSlotView(rewriter, loc, backings[index], tileType, nextOffset));
    }

    Value nextIv =
        arith::AddIOp::create(rewriter, loc, dbLoop.getInductionVar(),
                              dbLoop.getStep());
    Value hasNext =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, nextIv,
                              dbLoop.getUpperBound());
    auto prefetch = scf::IfOp::create(rewriter, loc, hasNext,
                                      /*withElseRegion=*/false);
    prefetch->setAttr(kPlanPrefetchAttr, UnitAttr::get(context));
    {
      OpBuilder::InsertionGuard ifGuard(rewriter);
      rewriter.setInsertionPointToStart(prefetch.thenBlock());
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), nextIv);
      for (auto [index, tile] : llvm::enumerate(plan.tiles))
        mapping.map(tile.root, nextViews[index]);
      for (TileInfo tile : plan.tiles) {
        for (Operation *setup : tile.setupOps)
          cloneMappedOp(setup, rewriter, mapping, sourceBlock,
                        /*markCompute=*/false);
        cloneCopy(tile.load, rewriter, mapping, sourceBlock, kPrefetchRole);
      }
    }

    IRMapping currentMapping;
    currentMapping.map(loop.getInductionVar(), dbLoop.getInductionVar());
    for (auto [index, tile] : llvm::enumerate(plan.tiles))
      currentMapping.map(tile.root, currentViews[index]);
    for (Operation *compute : plan.computes)
      cloneMappedOp(compute, rewriter, currentMapping, sourceBlock,
                    /*markCompute=*/true);
    for (memref::CopyOp store : plan.stores)
      cloneCopy(store, rewriter, currentMapping, sourceBlock, kDB2StoreRole);

    scf::YieldOp::create(rewriter, loc, next);
  }

  rewriter.eraseOp(loop);
  return true;
}

struct HexagonDoubleBufferPlanRewritePass
    : public ::impl::HexagonDoubleBufferPlanRewriteBase<
          HexagonDoubleBufferPlanRewritePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    int64_t nextPlanId = 0;

    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    for (scf::ForOp loop : loops) {
      auto kind = loop->getAttrOfType<StringAttr>(kLoopKindAttr);
      if (!kind || (kind.getValue() != kPointwise && kind.getValue() != kReduce))
        continue;

      Plan plan;
      plan.isReduction = kind.getValue() == kReduce;
      TileDataFlowAnalysis analysis(loop, aliasAnalysis);
      if (!analysis.run(plan))
        continue;

      IRRewriter rewriter(loop.getContext());
      rewritePointwisePlan(rewriter, plan, nextPlanId++);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferPlanRewritePass() {
  return std::make_unique<HexagonDoubleBufferPlanRewritePass>();
}

//===- HoistDoubleBufferCopyInsPass.cpp -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/CopyDirection.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HOISTDOUBLEBUFFERCOPYINS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

bool isGlobalToShared(memref::CopyOp copy) {
  auto direction = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return direction && direction.getValue() == kGlobalToShared;
}

bool isSharedToGlobal(memref::CopyOp copy) {
  auto direction = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return direction && direction.getValue() == kSharedToGlobal;
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

enum CopyFlowAccess : uint8_t {
  NoAccess = 0,
  ReadAccess = 1,
  WriteAccess = 2,
};

struct CopyFlowNode {
  Operation *op;
  StringRef role;
  SmallVector<uint8_t> accesses;
  SmallVector<unsigned> predecessors;
};

/// 为 copy-in 前移 pass 构建局部内存流图，用于展示转换前后的变化。
class CopyInHoistAnalysis {
public:
  CopyInHoistAnalysis(scf::ForOp forOp, AliasAnalysis &aliasAnalysis)
      : forOp(forOp), aliasAnalysis(aliasAnalysis) {}

  void build() {
    collectTiles();
    collectNodes();
    buildDependencies();
  }

  std::optional<unsigned> getNodeIndex(Operation *op) const {
    for (auto [index, node] : llvm::enumerate(nodes))
      if (node.op == op)
        return index;
    return std::nullopt;
  }

  void print(raw_ostream &os, StringRef title) const {
    os << "\n=== CopyInHoistAnalysis: " << title << " ===\n";
    os << "loop: ";
    forOp->print(os, OpPrintingFlags().skipRegions());
    os << "\n";

    os << "nodes:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      os << "  [" << nodeIndex << "] " << node.role << " accesses={";
      bool first = true;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        if (access == NoAccess)
          continue;
        if (!first)
          os << ", ";
        first = false;
        os << "tile#" << tileIndex << ":";
        if (access == (ReadAccess | WriteAccess))
          os << "READ_WRITE";
        else if (access == ReadAccess)
          os << "READ";
        else
          os << "WRITE";
      }
      os << "}\n";
      os << "      op: ";
      node.op->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }

    os << "dataflow:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      for (unsigned predecessor : node.predecessors) {
        const CopyFlowNode &source = nodes[predecessor];
        for (unsigned tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
          uint8_t sourceAccess = source.accesses[tileIndex];
          uint8_t targetAccess = node.accesses[tileIndex];
          bool raw =
              (sourceAccess & WriteAccess) && (targetAccess & ReadAccess);
          bool war =
              (sourceAccess & ReadAccess) && (targetAccess & WriteAccess);
          bool waw =
              (sourceAccess & WriteAccess) && (targetAccess & WriteAccess);
          if (!raw && !war && !waw)
            continue;
          os << "  [" << predecessor << "] " << source.role << " --";
          if (raw)
            os << "RAW";
          if (war)
            os << (raw ? "+WAR" : "WAR");
          if (waw)
            os << ((raw || war) ? "+WAW" : "WAW");
          os << "(tile#" << tileIndex << ")--> [" << nodeIndex << "] "
             << node.role << "\n";
        }
      }
    }
    os << "=== End CopyInHoistAnalysis ===\n";
  }

private:
  std::optional<unsigned> getTileIndex(Value value) const {
    if (!isa<BaseMemRefType>(value.getType()))
      return std::nullopt;
    for (auto [index, alloc] : llvm::enumerate(tiles))
      if (aliasAnalysis.alias(value, alloc->getResult(0)))
        return index;
    return std::nullopt;
  }

  void collectTiles() {
    tiles.clear();
    llvm::SmallDenseSet<Operation *> seen;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy || !isGlobalToShared(copy))
        continue;
      memref::AllocOp alloc = findBaseAlloc(copy.getTarget());
      if (alloc && seen.insert(alloc).second)
        tiles.push_back(alloc);
    }
  }

  void addPreciseEffects(Operation *op, CopyFlowNode &node) {
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return;
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      Value value = effect.getValue();
      if (!value)
        continue;
      auto tileIndex = getTileIndex(value);
      if (!tileIndex)
        continue;
      if (isa<MemoryEffects::Read>(effect.getEffect()))
        node.accesses[*tileIndex] |= ReadAccess;
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        node.accesses[*tileIndex] |= WriteAccess;
    }
  }

  void collectNodes() {
    nodes.clear();
    for (Operation &op : forOp.getBody()->without_terminator()) {
      CopyFlowNode node{
          &op, "COMPUTE", SmallVector<uint8_t>(tiles.size(), NoAccess), {}};
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        if (isGlobalToShared(copy))
          node.role = "LOAD";
        else if (isSharedToGlobal(copy))
          node.role = "STORE";
        else
          continue;
        if (auto source = getTileIndex(copy.getSource()))
          node.accesses[*source] |= ReadAccess;
        if (auto target = getTileIndex(copy.getTarget()))
          node.accesses[*target] |= WriteAccess;
      } else {
        addPreciseEffects(&op, node);
      }

      if (llvm::any_of(node.accesses,
                       [](uint8_t access) { return access != NoAccess; }))
        nodes.push_back(std::move(node));
    }
  }

  void buildDependencies() {
    SmallVector<std::optional<unsigned>> lastWrite(tiles.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tiles.size());
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      llvm::SmallDenseSet<unsigned> predecessors;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        bool reads = access & ReadAccess;
        bool writes = access & WriteAccess;
        if (reads && lastWrite[tileIndex])
          predecessors.insert(*lastWrite[tileIndex]);
        if (writes) {
          if (lastWrite[tileIndex])
            predecessors.insert(*lastWrite[tileIndex]);
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
        } else if (reads) {
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors);
    }
  }

  scf::ForOp forOp;
  AliasAnalysis &aliasAnalysis;
  SmallVector<memref::AllocOp> tiles;
  SmallVector<CopyFlowNode> nodes;
};

bool isMovableAddressOp(Operation *op) {
  return isMemoryEffectFree(op) ||
         isa<memref::AllocOp, memref::SubViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(op);
}

/// 收集 value 在当前 block 内、anchor 之后的定义链。返回顺序保证依赖在前。
bool collectMovableDefs(Value value, Block *block, Operation *anchor,
                        Operation *copy,
                        llvm::SetVector<Operation *> &movableOps) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != block || def->isBeforeInBlock(anchor))
    return true;
  if (def == copy || !def->isBeforeInBlock(copy) || !isMovableAddressOp(def))
    return false;

  for (Value operand : def->getOperands())
    if (!collectMovableDefs(operand, block, anchor, copy, movableOps))
      return false;
  movableOps.insert(def);
  return true;
}

/// 精确查询 op 的 memory effects。如果 effect 没有关联具体 Value，则认为
/// 可能冲突，避免错误前移。
bool hasInterveningConflict(Operation *op, Value source, Value target,
                            AliasAnalysis &aliasAnalysis) {
  if (isMemoryEffectFree(op))
    return false;

  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
    bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
    if (!isRead && !isWrite)
      continue;

    Value effectValue = effect.getValue();
    if (!effectValue)
      return true;

    // copy 前移时，不能跨越 source 的写，也不能跨越 target 的读或写。
    if (isWrite && aliasAnalysis.alias(effectValue, source))
      return true;
    if ((isRead || isWrite) && aliasAnalysis.alias(effectValue, target))
      return true;
  }
  return false;
}

bool hoistCopyBefore(memref::CopyOp copy, Operation *anchor,
                     AliasAnalysis &aliasAnalysis) {
  Block *block = copy->getBlock();
  llvm::SetVector<Operation *> movableOps;
  if (!collectMovableDefs(copy.getSource(), block, anchor, copy, movableOps) ||
      !collectMovableDefs(copy.getTarget(), block, anchor, copy, movableOps))
    return false;

  for (Operation *current = anchor; current && current != copy;
       current = current->getNextNode()) {
    if (movableOps.contains(current))
      continue;
    if (hasInterveningConflict(current, copy.getSource(), copy.getTarget(),
                               aliasAnalysis))
      return false;
  }

  // 依赖定义按拓扑顺序移动，最后移动 copy，保持 SSA dominance。
  for (Operation *op : movableOps)
    op->moveBefore(anchor);
  copy->moveBefore(anchor);
  return true;
}

void hoistCopyIns(scf::ForOp forOp, AliasAnalysis &aliasAnalysis) {
  Block *body = forOp.getBody();
  SmallVector<memref::CopyOp> copyIns;
  for (Operation &op : body->without_terminator())
    if (auto copy = dyn_cast<memref::CopyOp>(&op);
        copy && isGlobalToShared(copy))
      copyIns.push_back(copy);
  if (copyIns.size() < 2)
    return;

  // 第一处非地址计算、非 copy-in 的操作视为计算区起点。
  Operation *anchor = nullptr;
  for (Operation *op = copyIns.front()->getNextNode();
       op && op != body->getTerminator(); op = op->getNextNode()) {
    if (auto copy = dyn_cast<memref::CopyOp>(op);
        copy && isGlobalToShared(copy))
      continue;
    if (isMovableAddressOp(op))
      continue;
    anchor = op;
    break;
  }
  if (!anchor)
    return;

  CopyInHoistAnalysis before(forOp, aliasAnalysis);
  before.build();
  llvm::SmallDenseMap<Operation *, unsigned> oldPositions;
  for (memref::CopyOp copy : copyIns)
    if (auto index = before.getNodeIndex(copy))
      oldPositions[copy] = *index;

  SmallVector<Operation *> movedCopies;
  // 只处理原本位于计算区之后的 copy-in，并保持它们之间的原始顺序。
  for (memref::CopyOp copy : copyIns)
    if (anchor->isBeforeInBlock(copy) &&
        hoistCopyBefore(copy, anchor, aliasAnalysis))
      movedCopies.push_back(copy);

  if (movedCopies.empty())
    return;

  CopyInHoistAnalysis after(forOp, aliasAnalysis);
  after.build();
  before.print(llvm::outs(), "BEFORE");
  after.print(llvm::outs(), "AFTER");
  llvm::outs() << "moved copy-ins:\n";
  for (Operation *copy : movedCopies) {
    std::optional<unsigned> newPosition = after.getNodeIndex(copy);
    llvm::outs() << "  LOAD [" << oldPositions.lookup(copy) << "] -> ["
                 << (newPosition ? std::to_string(*newPosition) : "unknown")
                 << "]\n";
  }
}

struct HoistDoubleBufferCopyInsPass
    : public ::impl::HoistDoubleBufferCopyInsBase<
          HoistDoubleBufferCopyInsPass> {
  void runOnOperation() override {
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    getOperation().walk(
        [&](scf::ForOp forOp) { hoistCopyIns(forOp, aliasAnalysis); });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createHoistDoubleBufferCopyInsPass() {
  return std::make_unique<HoistDoubleBufferCopyInsPass>();
}

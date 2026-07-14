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

bool hasCopyDirection(memref::CopyOp copy, StringRef expected) {
  auto direction = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return direction && direction.getValue() == expected;
}

bool isLoadCopy(memref::CopyOp copy) {
  return hasCopyDirection(copy, kGlobalToShared);
}

bool isStoreCopy(memref::CopyOp copy) {
  return hasCopyDirection(copy, kSharedToGlobal);
}

bool isScheduledCopy(Operation *op) {
  auto copy = dyn_cast<memref::CopyOp>(op);
  return copy && (isLoadCopy(copy) || isStoreCopy(copy));
}

enum class ScheduleRole { Load, Store, Compute };

struct DataFlowAccess {
  unsigned tileIndex = 0;
  bool reads = false;
  bool writes = false;
};

struct DataFlowEdge {
  unsigned predecessor = 0;
  unsigned tileIndex = 0;
  bool raw = false;
  bool war = false;
  bool waw = false;
};

struct DataFlowNode {
  Operation *op = nullptr;
  ScheduleRole role = ScheduleRole::Compute;
  SmallVector<DataFlowAccess> accesses;
  SmallVector<DataFlowEdge> predecessors;
  bool hasUnknownMemoryEffect = false;
};

Value findMemoryRoot(Value value) {
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
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      value = view.getSource();
      continue;
    }

    return value;
  }

  return {};
}

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

StringRef stringifyAccess(bool reads, bool writes) {
  if (reads && writes)
    return "READ_WRITE";
  if (reads)
    return "READ";
  if (writes)
    return "WRITE";
  return "NO_ACCESS";
}

void addAccess(SmallVectorImpl<DataFlowAccess> &accesses, unsigned tileIndex,
               bool reads, bool writes) {
  for (DataFlowAccess &access : accesses) {
    if (access.tileIndex == tileIndex) {
      access.reads |= reads;
      access.writes |= writes;
      return;
    }
  }

  accesses.push_back(DataFlowAccess{tileIndex, reads, writes});
}

class CopyScheduleDataFlowAnalysis {
public:
  CopyScheduleDataFlowAnalysis(Block &block, AliasAnalysis &aliasAnalysis)
      : block(block), aliasAnalysis(aliasAnalysis) {}

  void build() {
    collectTiles();
    nodes.clear();
    opToNode.clear();
    for (Operation &op : block.without_terminator()) {
      std::optional<DataFlowNode> node = collectNode(&op);
      if (!node)
        continue;

      nodes.push_back(std::move(*node));
      opToNode.try_emplace(nodes.back().op, nodes.size() - 1);
    }
    buildDependencies();
  }

  SmallVector<memref::CopyOp> getLoads() const {
    SmallVector<memref::CopyOp> copies;
    for (const DataFlowNode &node : nodes)
      if (node.role == ScheduleRole::Load)
        copies.push_back(cast<memref::CopyOp>(node.op));
    return copies;
  }

  SmallVector<memref::CopyOp> getStores() const {
    SmallVector<memref::CopyOp> copies;
    for (const DataFlowNode &node : nodes)
      if (node.role == ScheduleRole::Store)
        copies.push_back(cast<memref::CopyOp>(node.op));
    return copies;
  }

  std::optional<unsigned> getNodeIndex(Operation *op) const {
    auto it = opToNode.find(op);
    if (it == opToNode.end())
      return std::nullopt;
    return it->second;
  }

  bool empty() const { return nodes.empty(); }

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
  void collectTiles() {
    tileRoots.clear();
    for (Operation &op : block.without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy)
        continue;

      Value tile = {};
      if (isLoadCopy(copy))
        tile = findMemoryRoot(copy.getTarget());
      else if (isStoreCopy(copy))
        tile = findMemoryRoot(copy.getSource());
      else
        continue;

      if (!tile || !tile.getDefiningOp<memref::AllocOp>())
        continue;
      if (!llvm::is_contained(tileRoots, tile))
        tileRoots.push_back(tile);
    }
  }

  std::optional<unsigned> getTileIndex(Value value) const {
    value = findMemoryRoot(value);
    if (!value)
      return std::nullopt;
    for (auto [index, tile] : llvm::enumerate(tileRoots))
      if (value == tile || aliasAnalysis.alias(value, tile))
        return index;
    return std::nullopt;
  }

  std::optional<DataFlowNode> collectNode(Operation *op) const {
    DataFlowNode node;
    node.op = op;

    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (isLoadCopy(copy))
        node.role = ScheduleRole::Load;
      else if (isStoreCopy(copy))
        node.role = ScheduleRole::Store;
      else
        return std::nullopt;

      if (auto source = getTileIndex(copy.getSource()))
        addAccess(node.accesses, *source, /*reads=*/true, /*writes=*/false);
      if (auto target = getTileIndex(copy.getTarget()))
        addAccess(node.accesses, *target, /*reads=*/false, /*writes=*/true);
      return node;
    }

    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return std::nullopt;

    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      bool reads = isa<MemoryEffects::Read>(effect.getEffect());
      bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (!reads && !writes)
        continue;

      Value value = effect.getValue();
      if (!value) {
        node.hasUnknownMemoryEffect = true;
        continue;
      }
      if (auto tileIndex = getTileIndex(value))
        addAccess(node.accesses, *tileIndex, reads, writes);
    }

    if (node.accesses.empty() && !node.hasUnknownMemoryEffect)
      return std::nullopt;
    return node;
  }

  void buildDependencies() {
    SmallVector<std::optional<unsigned>> lastWrite(tileRoots.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tileRoots.size());

    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      SmallVector<DataFlowEdge> edges;
      for (const DataFlowAccess &access : node.accesses) {
        bool reads = access.reads;
        bool writes = access.writes;
        unsigned tileIndex = access.tileIndex;

        if (reads && lastWrite[tileIndex])
          addOrMergeEdge(edges, *lastWrite[tileIndex], tileIndex,
                         /*raw=*/true, /*war=*/false, /*waw=*/false);

        if (writes) {
          if (lastWrite[tileIndex])
            addOrMergeEdge(edges, *lastWrite[tileIndex], tileIndex,
                           /*raw=*/false, /*war=*/false, /*waw=*/true);
          for (unsigned read : readsSinceWrite[tileIndex])
            addOrMergeEdge(edges, read, tileIndex, /*raw=*/false,
                           /*war=*/true, /*waw=*/false);
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
          if (reads)
            readsSinceWrite[tileIndex].push_back(nodeIndex);
        } else if (reads) {
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      llvm::sort(edges, [](const DataFlowEdge &lhs, const DataFlowEdge &rhs) {
        if (lhs.predecessor != rhs.predecessor)
          return lhs.predecessor < rhs.predecessor;
        return lhs.tileIndex < rhs.tileIndex;
      });
      node.predecessors = std::move(edges);
    }
  }

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

  Block &block;
  AliasAnalysis &aliasAnalysis;
  SmallVector<Value> tileRoots;
  SmallVector<DataFlowNode> nodes;
  llvm::DenseMap<Operation *, unsigned> opToNode;
};

bool isMovableAddressOp(Operation *op) {
  return op->getNumRegions() == 0 &&
         (isMemoryEffectFree(op) ||
          isa<memref::AllocOp, memref::SubViewOp, memref::ViewOp,
              memref::CastOp, memref::ReinterpretCastOp>(op));
}

bool collectMovableDefs(Value value, Block *block, Operation *anchor,
                        Operation *copy,
                        llvm::SetVector<Operation *> &movableOps) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != block || def->isBeforeInBlock(anchor))
    return true;
  if (def == anchor || def == copy || !def->isBeforeInBlock(copy) ||
      !isMovableAddressOp(def))
    return false;

  for (Value operand : def->getOperands())
    if (!collectMovableDefs(operand, block, anchor, copy, movableOps))
      return false;

  movableOps.insert(def);
  return true;
}

bool hasConflictWhenCrossing(Operation *op, Value source, Value target,
                             AliasAnalysis &aliasAnalysis) {
  if (isScheduledCopy(op))
    return true;

  source = findMemoryRoot(source);
  target = findMemoryRoot(target);

  if (isMemoryEffectFree(op))
    return false;

  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    bool reads = isa<MemoryEffects::Read>(effect.getEffect());
    bool writes = isa<MemoryEffects::Write>(effect.getEffect());
    if (!reads && !writes)
      continue;

    Value effectValue = effect.getValue();
    if (!effectValue)
      return true;
    effectValue = findMemoryRoot(effectValue);

    if (writes &&
        (effectValue == source || aliasAnalysis.alias(effectValue, source)))
      return true;
    if ((reads || writes) &&
        (effectValue == target || aliasAnalysis.alias(effectValue, target)))
      return true;
  }

  return false;
}

bool hoistLoadBefore(memref::CopyOp copy, Operation *anchor,
                     AliasAnalysis &aliasAnalysis) {
  if (copy == anchor || anchor->getBlock() != copy->getBlock() ||
      !anchor->isBeforeInBlock(copy))
    return false;

  Block *block = copy->getBlock();
  llvm::SetVector<Operation *> movableOps;
  if (!collectMovableDefs(copy.getSource(), block, anchor, copy, movableOps) ||
      !collectMovableDefs(copy.getTarget(), block, anchor, copy, movableOps))
    return false;

  for (Operation *current = anchor; current && current != copy;
       current = current->getNextNode()) {
    if (movableOps.contains(current))
      continue;
    if (hasConflictWhenCrossing(current, copy.getSource(), copy.getTarget(),
                                aliasAnalysis))
      return false;
  }

  for (Operation *op : movableOps)
    op->moveBefore(anchor);
  copy->moveBefore(anchor);
  return true;
}

bool hoistLoadAsEarlyAsPossible(memref::CopyOp copy,
                                AliasAnalysis &aliasAnalysis) {
  SmallVector<Operation *> anchors;
  for (Operation *op = &copy->getBlock()->front(); op && op != copy;
       op = op->getNextNode())
    anchors.push_back(op);

  for (Operation *anchor : anchors)
    if (anchor->getBlock() == copy->getBlock() &&
        anchor->isBeforeInBlock(copy) &&
        hoistLoadBefore(copy, anchor, aliasAnalysis))
      return true;

  return false;
}

bool sinkStoreAfter(memref::CopyOp copy, Operation *insertAfter,
                    AliasAnalysis &aliasAnalysis) {
  if (copy == insertAfter || insertAfter->getBlock() != copy->getBlock() ||
      !copy->isBeforeInBlock(insertAfter))
    return false;

  for (Operation *current = copy->getNextNode(); current && current != nullptr;
       current = current->getNextNode()) {
    if (current == insertAfter)
      break;
    if (hasConflictWhenCrossing(current, copy.getSource(), copy.getTarget(),
                                aliasAnalysis))
      return false;
  }
  if (hasConflictWhenCrossing(insertAfter, copy.getSource(), copy.getTarget(),
                              aliasAnalysis))
    return false;

  copy->moveBefore(insertAfter->getNextNode());
  return true;
}

bool sinkStoreAsLateAsPossible(memref::CopyOp copy,
                               AliasAnalysis &aliasAnalysis) {
  Operation *terminator = copy->getBlock()->getTerminator();
  Operation *lastLegal = nullptr;
  for (Operation *op = copy->getNextNode(); op && op != terminator;
       op = op->getNextNode()) {
    if (hasConflictWhenCrossing(op, copy.getSource(), copy.getTarget(),
                                aliasAnalysis))
      break;
    lastLegal = op;
  }

  if (!lastLegal)
    return false;
  return sinkStoreAfter(copy, lastLegal, aliasAnalysis);
}

bool scheduleBlock(Block &block, AliasAnalysis &aliasAnalysis) {
  if (!block.getTerminator())
    return false;

  bool changed = false;
  bool localChanged = true;
  CopyScheduleDataFlowAnalysis before(block, aliasAnalysis);
  before.build();
  before.print(llvm::outs(), "BEFORE");

  while (localChanged) {
    localChanged = false;

    CopyScheduleDataFlowAnalysis analysis(block, aliasAnalysis);
    analysis.build();

    for (memref::CopyOp copy : analysis.getLoads()) {
      if (copy->getBlock() != &block)
        continue;
      if (hoistLoadAsEarlyAsPossible(copy, aliasAnalysis)) {
        localChanged = true;
        changed = true;
      }
    }

    analysis.build();

    for (memref::CopyOp copy : llvm::reverse(analysis.getStores())) {
      if (copy->getBlock() != &block)
        continue;
      if (sinkStoreAsLateAsPossible(copy, aliasAnalysis)) {
        localChanged = true;
        changed = true;
      }
    }
  }

  CopyScheduleDataFlowAnalysis after(block, aliasAnalysis);
  after.build();
  after.print(llvm::outs(), "AFTER");

  return changed;
}

struct ScheduleDoubleBufferCopiesPass
    : public ::impl::ScheduleDoubleBufferCopiesBase<
          ScheduleDoubleBufferCopiesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kLoopKindAttr))
        loops.push_back(loop);
    });

    for (scf::ForOp loop : loops)
      scheduleBlock(*loop.getBody(), aliasAnalysis);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createScheduleDoubleBufferCopiesPass() {
  return std::make_unique<ScheduleDoubleBufferCopiesPass>();
}

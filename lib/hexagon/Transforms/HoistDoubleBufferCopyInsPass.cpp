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

/// 沿着一个 memref value 追溯到底层 alloc
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
  // 无访问、读、写
  NoAccess = 0,
  ReadAccess = 1,
  WriteAccess = 2,
};

struct CopyFlowNode {
  // op、角色、对各 tile 的访问、前驱节点
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
    // 依次收集 tile、节点、依赖边
    collectTiles();
    collectNodes();
    buildDependencies();
  }

  // 根据 Operation* 查找它在 nodes 中的编号
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
  /// 判断一个 memref value 别名于哪个 tile alloc
  std::optional<unsigned> getTileIndex(Value value) const {
    // 非 memref 类型直接返回空
    if (!isa<BaseMemRefType>(value.getType()))
      return std::nullopt;
    // 遍历 tiles，用 aliasAnalysis 判断是否别名
    for (auto [index, alloc] : llvm::enumerate(tiles))
      if (aliasAnalysis.alias(value, alloc->getResult(0)))
        return index;
    // 找不到对应 tile 返回空
    return std::nullopt;
  }

  /// 从 global_to_shared copy 的 target 中找 shared tile
  void collectTiles() {
    // 清空旧 tiles
    tiles.clear();
    // seen 用来去重 alloc
    llvm::SmallDenseSet<Operation *> seen;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      // 不是 copy-in 则跳过
      if (!copy || !isGlobalToShared(copy))
        continue;
      // 从 copy target 追溯底层 alloc
      memref::AllocOp alloc = findBaseAlloc(copy.getTarget());
      // 找到且未见过则加入 tiles
      if (alloc && seen.insert(alloc).second)
        tiles.push_back(alloc);
    }
  }

  /// op 的 memory effects 映射到 tile 读写
  void addPreciseEffects(Operation *op, CopyFlowNode &node) {
    // 如果 op 不支持 MemoryEffectOpInterface，就无法精确分析，直接不记录
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return;
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      Value value = effect.getValue();
      // 没有关联 value 的 effect 跳过
      if (!value)
        continue;
      // 判断 effect value 属于哪个 tile；不属于则跳过
      auto tileIndex = getTileIndex(value);
      if (!tileIndex)
        continue;
      // Read effect 标记 READ
      if (isa<MemoryEffects::Read>(effect.getEffect()))
        node.accesses[*tileIndex] |= ReadAccess;
      // Write effect 标记 WRITE
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        node.accesses[*tileIndex] |= WriteAccess;
    }
  }

  /// 收集 LOAD/STORE/COMPUTE 数据流节点
  void collectNodes() {
    // 清空旧 nodes
    nodes.clear();
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // 默认创建 COMPUTE 节点，访问数组长度等于 tile 数
      CopyFlowNode node{
          &op, "COMPUTE", SmallVector<uint8_t>(tiles.size(), NoAccess), {}};
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        if (isGlobalToShared(copy))
          // global_to_shared 标为 LOAD
          node.role = "LOAD";
        else if (isSharedToGlobal(copy))
          // shared_to_global 标为 STORE
          node.role = "STORE";
        else
          // 其他 copy 不参与本分析，跳过
          continue;
        // 如果 copy source 是某个 tile，标记 READ
        if (auto source = getTileIndex(copy.getSource()))
          node.accesses[*source] |= ReadAccess;
        // 如果 copy target 是某个 tile，标记 WRITE
        if (auto target = getTileIndex(copy.getTarget()))
          node.accesses[*target] |= WriteAccess;
      } else {
        // 非 copy op 则通过 memory effect 分析其读写
        addPreciseEffects(&op, node);
      }

      // 只保留确实访问了某个 tile 的节点
      if (llvm::any_of(node.accesses,
                       [](uint8_t access) { return access != NoAccess; }))
        nodes.push_back(std::move(node));
    }
  }

  /// 根据每个 tile 的读写顺序构造依赖前驱
  void buildDependencies() {
    // lastWrite 记录每个 tile 最近一次写节点
    SmallVector<std::optional<unsigned>> lastWrite(tiles.size());
    // readsSinceWrite 记录每个 tile 最近一次写之后的读节点集合
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tiles.size());
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      llvm::SmallDenseSet<unsigned> predecessors;
      for (auto [tileIndex, access] : llvm::enumerate(node.accesses)) {
        // 计算是否读、是否写
        bool reads = access & ReadAccess;
        bool writes = access & WriteAccess;
        // 读必须依赖最近一次写，形成 RAW
        if (reads && lastWrite[tileIndex])
          predecessors.insert(*lastWrite[tileIndex]);
        if (writes) {
          // 写依赖最近一次写，形成 WAW
          if (lastWrite[tileIndex])
            predecessors.insert(*lastWrite[tileIndex]);
          // 写也依赖最近一次写之后的读，形成 WAR
          predecessors.insert(readsSinceWrite[tileIndex].begin(),
                              readsSinceWrite[tileIndex].end());
          // 写发生后，清空 read 集合
          readsSinceWrite[tileIndex].clear();
          // 更新最近一次写
          lastWrite[tileIndex] = nodeIndex;
        } else if (reads) {
          // 只读，把当前节点记入 readsSinceWrite
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      // 把前驱集合写入 node 并排序，保证输出稳定
      node.predecessors.assign(predecessors.begin(), predecessors.end());
      llvm::sort(node.predecessors);
    }
  }

  scf::ForOp forOp;
  AliasAnalysis &aliasAnalysis;
  SmallVector<memref::AllocOp> tiles;
  SmallVector<CopyFlowNode> nodes;
};

/// 判断 op 是否可以作为地址计算一起移动
bool isMovableAddressOp(Operation *op) {
  // memory-effect-free 的 op 可移动
  // alloc/subview/cast/reinterpret_cast 也视为可移动地址 op
  return isMemoryEffectFree(op) ||
         isa<memref::AllocOp, memref::SubViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(op);
}

/// 收集 value 在当前 block 内、anchor 之后的定义链。返回顺序保证依赖在前。
bool collectMovableDefs(Value value, Block *block, Operation *anchor,
                        Operation *copy,
                        llvm::SetVector<Operation *> &movableOps) {
  Operation *def = value.getDefiningOp();

  // 没有定义、定义不在当前 block、或定义已在 anchor 前，则无需移动
  if (!def || def->getBlock() != block || def->isBeforeInBlock(anchor))
    return true;

  // 如果定义就是 copy、本身不在 copy 前、或不可移动，则失败
  if (def == copy || !def->isBeforeInBlock(copy) || !isMovableAddressOp(def))
    return false;

  // 递归处理定义 op 的 operands，保证依赖先于使用者
  for (Value operand : def->getOperands())
    if (!collectMovableDefs(operand, block, anchor, copy, movableOps))
      return false;
  // 把当前定义 op 加入 SetVector
  movableOps.insert(def);
  return true;
}

/// 查询 op 的 memory effects。如果 effect 没有关联具体 Value，则认为
/// 可能冲突，避免错误前移。
bool hasInterveningConflict(Operation *op, Value source, Value target,
                            AliasAnalysis &aliasAnalysis) {
  // 无副作用 op 不冲突
  if (isMemoryEffectFree(op))
    return false;

  // 不支持 MemoryEffectOpInterface 的 op 保守认为冲突
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    // 判断 effect 是读还是写
    bool isRead = isa<MemoryEffects::Read>(effect.getEffect());
    bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());

    // 非读写 effect 忽略
    if (!isRead && !isWrite)
      continue;

    // effect 没有关联具体 value，保守认为冲突
    Value effectValue = effect.getValue();
    if (!effectValue)
      return true;

    // copy 前移时，不能跨越 source 的写，也不能跨越 target 的读或写。
    if (isWrite && aliasAnalysis.alias(effectValue, source))
      return true;
    if ((isRead || isWrite) && aliasAnalysis.alias(effectValue, target))
      return true;
  }
  // 所有 effect 都安全则无冲突
  return false;
}

/// 尝试把一个 copy 移到 anchor 前
bool hoistCopyBefore(memref::CopyOp copy, Operation *anchor,
                     AliasAnalysis &aliasAnalysis) {
  Block *block = copy->getBlock();
  llvm::SetVector<Operation *> movableOps;
  // 收集 source 和 target 的 movable defs；失败则不移动
  if (!collectMovableDefs(copy.getSource(), block, anchor, copy, movableOps) ||
      !collectMovableDefs(copy.getTarget(), block, anchor, copy, movableOps))
    return false;

  // 遍历 anchor 到 copy 之间的 op，检查是否存在冲突
  for (Operation *current = anchor; current && current != copy;
       current = current->getNextNode()) {
    // 如果当前 op 是准备一起移动的地址定义，跳过冲突检查
    if (movableOps.contains(current))
      continue;
    // 有冲突则放弃移动
    if (hasInterveningConflict(current, copy.getSource(), copy.getTarget(),
                               aliasAnalysis))
      return false;
  }

  /// 按 SetVector 顺序把依赖定义移到 anchor 前
  for (Operation *op : movableOps)
    op->moveBefore(anchor);
  // 把 copy 自身移到 anchor 前
  copy->moveBefore(anchor);
  return true;
}

void hoistCopyIns(scf::ForOp forOp, AliasAnalysis &aliasAnalysis) {
  Block *body = forOp.getBody();
  SmallVector<memref::CopyOp> copyIns;
  // 遍历 body，收集 global_to_shared memref.copy
  for (Operation &op : body->without_terminator())
    if (auto copy = dyn_cast<memref::CopyOp>(&op);
        copy && isGlobalToShared(copy))
      copyIns.push_back(copy);
  if (copyIns.size() < 2)
    return;

  // 从第一个 copy-in 后面开始扫描
  // 遇到 copy-in 继续跳过
  // 遇到可移动地址 op 继续跳过
  // 第一个其他 op 作为 anchor，然后停止
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

  // 构建移动前的数据流分析
  CopyInHoistAnalysis before(forOp, aliasAnalysis);
  before.build();
  llvm::SmallDenseMap<Operation *, unsigned> oldPositions;
  // 保存每个 copy-in 的旧位置
  for (memref::CopyOp copy : copyIns)
    if (auto index = before.getNodeIndex(copy))
      oldPositions[copy] = *index;

  SmallVector<Operation *> movedCopies;
  // 只尝试移动原本在 anchor 后面的 copy-in，并保持遍历顺序
  for (memref::CopyOp copy : copyIns)
    if (anchor->isBeforeInBlock(copy) &&
        hoistCopyBefore(copy, anchor, aliasAnalysis))
      movedCopies.push_back(copy);

  if (movedCopies.empty())
    return;

  // 构建移动后的数据流分析
  CopyInHoistAnalysis after(forOp, aliasAnalysis);
  after.build();

  // 打印 before/after 数据流图
  before.print(llvm::outs(), "BEFORE");
  after.print(llvm::outs(), "AFTER");

  // 打印每个移动 copy 的旧位置到新位置
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
    // 获取 AliasAnalysis
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    // 遍历当前 func 内所有 scf.for，对每个调用 hoistCopyIns
    getOperation().walk(
        [&](scf::ForOp forOp) { hoistCopyIns(forOp, aliasAnalysis); });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createHoistDoubleBufferCopyInsPass() {
  return std::make_unique<HoistDoubleBufferCopyInsPass>();
}

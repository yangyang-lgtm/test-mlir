//===- ScheduleDoubleBufferLoadStoreExtPass.cpp --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h" // 本 pass 直接识别并生成 memref_ext 的 load/store/DMA 相关操作。
#include "hexagon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // 需要处理 tt.addptr 这类 Triton 指针计算，追溯真实内存来源。

#include "mlir/Analysis/AliasAnalysis.h" // 需要别名分析来判断 tile 根内存是否可能互相影响。
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h" // 需要读取操作的内存副作用，才能建立安全的数据依赖。

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_SCHEDULEDOUBLEBUFFERLOADSTOREEXT
#include "hexagon/Transforms/Passes.h.inc"

namespace { // 这些工具函数只服务本 pass，不暴露到链接符号表。

enum class ExtScheduleRole { Load, Store, Compute }; // 把操作分成输入预取、输出写回和计算三类，后续用不同移动策略处理。

struct ExtAccess { // 用 tile 维度记录读写集合，避免只靠操作顺序做粗粒度屏障。
  unsigned tileIndex = 0;
  bool reads = false;
  bool writes = false;
};

struct ExtEdge { // 依赖边保存前驱节点和依赖类型，决定移动不能跨过的位置。
  unsigned predecessor = 0;
  unsigned tileIndex = 0;
  bool raw = false; // RAW 表示当前读依赖前序写，load/store 移动不能破坏它。
  bool war = false; // WAR 表示后续写不能越过此前读，防止读到新值。
  bool waw = false; // WAW 表示多个写必须保持最终覆盖顺序。
};

struct ExtNode { // 节点把一个 IR 操作抽象成可调度的数据流单元。
  Operation *op = nullptr;
  ExtScheduleRole role = ExtScheduleRole::Compute;
  SmallVector<ExtAccess> accesses;
  SmallVector<ExtEdge> predecessors;
  bool hasUnknownMemoryEffect = false; // 未知副作用要当成访问所有 tile，避免错误重排。
};

struct ExtMove { // 移动计划把待移动 op、插入点和必须一起移动的地址定义放在一起。
  Operation *op = nullptr;
  Operation *insertionPoint = nullptr; // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
  SmallVector<Operation *> movableDefs; // 地址/size 计算如果也在移动区间内，必须跟随 load 一起前移。
};

bool isLoadExt(Operation *op) {
  return isa<memref_ext::LoadExOp, memref_ext::BlockDescriptorLoadOp,
             memref_ext::BlockDescLoadOp>(op);
}
bool isStoreExt(Operation *op) {
  return isa<memref_ext::StoreExOp, memref_ext::BlockDescriptorStoreOp,
             memref_ext::BlockDescStoreOp>(op);
}

Value getTransferSource(Operation *op) {
  if (auto load = dyn_cast<memref_ext::LoadExOp>(op))
    return load.getPtr();
  if (auto load = dyn_cast<memref_ext::BlockDescriptorLoadOp>(op))
    return load.getDesc();
  if (auto load = dyn_cast<memref_ext::BlockDescLoadOp>(op))
    return load.getDesc();
  if (auto store = dyn_cast<memref_ext::StoreExOp>(op))
    return store.getValue();
  if (auto store = dyn_cast<memref_ext::BlockDescriptorStoreOp>(op))
    return store.getValue();
  if (auto store = dyn_cast<memref_ext::BlockDescStoreOp>(op))
    return store.getValue();
  return {};
}

Value getTransferTarget(Operation *op) {
  if (auto load = dyn_cast<memref_ext::LoadExOp>(op))
    return load.getTarget();
  if (auto load = dyn_cast<memref_ext::BlockDescriptorLoadOp>(op))
    return load.getTarget();
  if (auto load = dyn_cast<memref_ext::BlockDescLoadOp>(op))
    return load.getTarget();
  if (auto store = dyn_cast<memref_ext::StoreExOp>(op))
    return store.getPtr();
  if (auto store = dyn_cast<memref_ext::BlockDescriptorStoreOp>(op))
    return store.getDesc();
  if (auto store = dyn_cast<memref_ext::BlockDescStoreOp>(op))
    return store.getDesc();
  return {};
}

bool isCopyLikeRole(ExtScheduleRole role) { // load 和 store 都是传输类操作，彼此之间保持顺序能避免 DMA 流水混乱。
  return role == ExtScheduleRole::Load || role == ExtScheduleRole::Store;
}

StringRef stringifyRole(ExtScheduleRole role) { // 调试图需要稳定的角色名，便于观察调度前后变化。
  switch (role) { // 按节点角色分派，保证调试输出和调度语义一致。
  case ExtScheduleRole::Load:
    return "LOAD_EXT";
  case ExtScheduleRole::Store:
    return "STORE_EXT";
  case ExtScheduleRole::Compute:
    return "COMPUTE";
  }
  llvm_unreachable("unknown schedule role"); // 枚举值都已覆盖，走到这里说明角色集合被错误扩展。
}

StringRef stringifyAccess(bool reads, bool writes) { // 调试图展示每个 tile 的访问模式，帮助确认依赖构建是否正确。
  if (reads && writes) // 调试图要显示该节点既消费旧 tile 又覆盖新 tile，便于核对 RAW/WAR/WAW 都被建出。
    return "READ_WRITE";
  if (reads) // 纯读节点不会产生新版本，只需要在调试输出中标为 READ。
    return "READ";
  if (writes) // 纯写节点会更新 tile 版本，调试输出中标为 WRITE 以对应 WAW/WAR 边。
    return "WRITE";
  return "NO_ACCESS";
}

std::string describeDependency(const ExtEdge &edge) { // 一条边可能同时承载多种依赖，组合输出能暴露真实约束。
  SmallVector<StringRef> kinds; // 临时收集依赖类型，后面按固定顺序拼成可读文本。
  if (edge.raw) // 这条边表示当前节点读到前驱写出的 tile，调试输出需要显式标出先写后读的约束。
    kinds.push_back("RAW"); // 把当前命中的依赖类型加入调试描述。
  if (edge.war) // 这条边表示前驱必须先读旧 tile，当前写不能提前覆盖它要消费的数据。
    kinds.push_back("WAR"); // 把当前命中的依赖类型加入调试描述。
  if (edge.waw) // 这条边表示两个节点写同一 tile，后写必须保持在后面才能保留最终内存状态。
    kinds.push_back("WAW"); // 把当前命中的依赖类型加入调试描述。

  if (kinds.empty()) // 没有可调度节点时继续优化没有意义。
    return "UNKNOWN";

  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::interleave(kinds, os, "+"); // 用加号连接依赖类型，明确一条边包含多个约束。
  return result;
}

Value findMemoryRoot(Value value) { // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
  while (value) { // 沿着等价内存值链向上追溯，直到找到能用于依赖判断的根。
    Operation *def = value.getDefiningOp();
    if (!def) // block 参数或函数参数本身就是可见内存根，再向上追溯没有定义 op 可走。
      return value;

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

    return value;
  }
  return {}; // 找不到可信根内存时返回空值，让调用方走保守路径。
}

void addAccess(SmallVectorImpl<ExtAccess> &accesses, unsigned tileIndex,
               bool reads, bool writes) {
  for (ExtAccess &access : accesses) {
    if (access.tileIndex == tileIndex) { // 一个 op 可能通过多个 effect 命中同一 tile，这里合并读写标志以形成单一访问摘要。
      access.reads |= reads;
      access.writes |= writes;
      return;
    }
  }
  accesses.push_back(ExtAccess{tileIndex, reads, writes});
}

bool isMovableAddressOp(Operation *op) { // load 前移时只允许纯地址计算一起移动，避免把有语义副作用的 op 提前。
  return op->getNumRegions() == 0 &&
         (isMemoryEffectFree(op) ||
          isa<memref::AllocOp, memref::SubViewOp, memref::ViewOp,
              memref::CastOp, memref::ReinterpretCastOp>(op));
}

bool collectMovableDefs(Value value, Block *block, Operation *anchor, // 递归收集 load 使用的本地定义，保证前移后操作数仍然支配 load。
                        Operation *scheduledOp,
                        llvm::SetVector<Operation *> &movableOps) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != block || def->isBeforeInBlock(anchor)) // 外部定义或已在锚点之前的定义已经支配新 load 位置，不需要跟随移动。
    return true;

  if (def == anchor || def == scheduledOp || !def->isBeforeInBlock(scheduledOp) || // 插入点没有带来真实位置变化时跳过，避免无意义改写。
      !isMovableAddressOp(def)) // load 前移时只允许纯地址计算一起移动，避免把有语义副作用的 op 提前。
    return false;

  for (Value operand : def->getOperands())
    if (!collectMovableDefs(operand, block, anchor, scheduledOp, movableOps)) // 任一操作数定义不能安全前移时，整个地址计算链都不能跟随 load 前移。
      return false;

  movableOps.insert(def); // 依赖先收集进有序集合，移动时按依赖顺序插到 load 前面。
  return true;
}

bool hasConflictWhenCrossing(Operation *op, Value source, Value target, // 移动跨过中间操作前，必须证明中间操作不读写相关源/目标内存。
                             AliasAnalysis &aliasAnalysis) {
  source = findMemoryRoot(source); // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
  target = findMemoryRoot(target); // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
  auto isAliasTracked = [](Value value) {
    return value &&
           (isa<BaseMemRefType>(value.getType()) ||
            isa<triton::PointerType>(value.getType()));
  };
  bool checkSource = isAliasTracked(source);
  bool checkTarget = isAliasTracked(target);

  if (isMemoryEffectFree(op)) // 纯计算不读写内存，load/store 跨过它不会改变源、目标或 tile 内容。
    return false;

  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp) // 没有 MemoryEffectOpInterface 时无法证明它不碰 source/target，跨越移动必须把它当作屏障。
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects); // 读取精确副作用列表，尽量少把无关操作当成移动屏障。
  for (const MemoryEffects::EffectInstance &effect : effects) {
    bool reads = isa<MemoryEffects::Read>(effect.getEffect());
    bool writes = isa<MemoryEffects::Write>(effect.getEffect());
    if (!reads && !writes) // 只关心 Read/Write；alloc/free 等效果不表示这里要保护的内存值顺序。
      continue;

    Value effectValue = effect.getValue(); // 副作用必须有具体内存值，否则无法证明与 tile 无关。
    if (!effectValue) // 没有具体 effect value 时无法做别名排除，必须认为它可能碰到 source 或 target。
      return true;
    effectValue = findMemoryRoot(effectValue); // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。

    if (checkSource && writes && // 写操作会结束此前读者的开放区间，并成为新的最近写。
        (effectValue == source || aliasAnalysis.alias(effectValue, source)))
      return true;
    if (checkTarget && (reads || writes) && // 写操作会结束此前读者的开放区间，并成为新的最近写。
        (effectValue == target || aliasAnalysis.alias(effectValue, target)))
      return true;
  }

  return false;
}

class LoadStoreExtDataFlowAnalysis {
public:
  LoadStoreExtDataFlowAnalysis(Block &block, AliasAnalysis &aliasAnalysis)
      : block(block), aliasAnalysis(aliasAnalysis) {}

  void build() {
    collectTiles();
    nodes.clear(); // 新的写会截断旧读区间，因此清掉已处理的读者集合。
    opToNode.clear(); // 新的写会截断旧读区间，因此清掉已处理的读者集合。
    for (Operation &op : block.without_terminator()) { // 只分析 loop body 的实际操作，terminator 不参与 tile 数据流。
      std::optional<ExtNode> node = collectNode(&op);
      if (!node) // 没有可建模 tile 访问的 op 不会影响 load/store_ext 的相对位置，跳过可减少无关依赖。
        continue;
      nodes.push_back(std::move(*node));
      opToNode.try_emplace(nodes.back().op, nodes.size() - 1);
    }
    buildDependencies();
  }

  SmallVector<Operation *> getLoads() const {
    SmallVector<Operation *> loads;
    for (const ExtNode &node : nodes)
      if (node.role == ExtScheduleRole::Load) // 调度阶段只对 load_ex 做前移优化，compute/store 节点不能混入这个候选列表。
        loads.push_back(node.op);
    return loads;
  }

  SmallVector<Operation *> getStores() const {
    SmallVector<Operation *> stores;
    for (const ExtNode &node : nodes)
      if (node.role == ExtScheduleRole::Store) // 调度阶段只对 store_ex 做后移优化，load/compute 节点不能混入这个候选列表。
        stores.push_back(node.op);
    return stores;
  }

  void print(raw_ostream &os, StringRef title) const {
    os << "\n=== ScheduleDoubleBufferLoadStoreExt dataflow: " << title
       << " ===\n";
    os << "loop: ";
    if (Operation *parent = block.getParentOp()) // 有父 op 时打印去掉 region 的循环壳，调试依赖图时能对应到具体 loop。
      parent->print(os, OpPrintingFlags().skipRegions());
    else
      os << "<unknown>";
    os << "\n";

    os << "nodes:\n";
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      os << "  [" << nodeIndex << "] " << stringifyRole(node.role); // 调试图需要稳定的角色名，便于观察调度前后变化。
      if (node.hasUnknownMemoryEffect) // 未知副作用会扩展为读写所有 tile，调试输出要提醒它为何限制了移动空间。
        os << " unknown-memory-effect";
      os << " accesses={";
      if (node.accesses.empty()) { // 没有可调度节点时继续优化没有意义。
        os << "}";
      } else {
        llvm::interleaveComma(node.accesses, os, // 用加号连接依赖类型，明确一条边包含多个约束。
                              [&](const ExtAccess &access) {
                                os << "tile#" << access.tileIndex << ":"
                                   << stringifyAccess(access.reads, // 调试图展示每个 tile 的访问模式，帮助确认依赖构建是否正确。
                                                      access.writes);
                              });
        os << "}";
      }
      os << "\n";

      if (auto load = dyn_cast<memref_ext::LoadExOp>(node.op)) { // load_ex 是输入预取点，它写入本地 tile，需要作为调度边界分析。
        os << "      load_ext memory: ptr="
           << findMemoryRoot(load.getPtr()) << ", target=" // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
           << findMemoryRoot(load.getTarget()) << "\n"; // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
      } else if (auto load =
                     dyn_cast<memref_ext::BlockDescriptorLoadOp>(node.op)) {
        os << "      block_descriptor_load memory: desc="
           << findMemoryRoot(load.getDesc()) << ", target="
           << findMemoryRoot(load.getTarget()) << "\n";
      } else if (auto load = dyn_cast<memref_ext::BlockDescLoadOp>(node.op)) {
        os << "      block_desc_load memory: desc="
           << findMemoryRoot(load.getDesc()) << ", target="
           << findMemoryRoot(load.getTarget()) << "\n";
      } else if (auto store = dyn_cast<memref_ext::StoreExOp>(node.op)) {
        os << "      store_ext memory: value="
           << findMemoryRoot(store.getValue()) << ", ptr=" // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
           << findMemoryRoot(store.getPtr()) << "\n"; // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
      } else if (auto store =
                     dyn_cast<memref_ext::BlockDescriptorStoreOp>(node.op)) {
        os << "      block_descriptor_store memory: value="
           << findMemoryRoot(store.getValue()) << ", desc="
           << findMemoryRoot(store.getDesc()) << "\n";
      } else if (auto store = dyn_cast<memref_ext::BlockDescStoreOp>(node.op)) {
        os << "      block_desc_store memory: value="
           << findMemoryRoot(store.getValue()) << ", desc="
           << findMemoryRoot(store.getDesc()) << "\n";
      }
      os << "      op: ";
      node.op->print(os, OpPrintingFlags().skipRegions());
      os << "\n";
    }

    os << "dataflow:\n";
    bool printedDependency = false;
    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      for (const ExtEdge &edge : node.predecessors) {
        printedDependency = true;
        const ExtNode &predecessor = nodes[edge.predecessor];
        os << "  [" << edge.predecessor << "] "
           << stringifyRole(predecessor.role) << " --"
           << describeDependency(edge) << "(tile#" << edge.tileIndex
           << ")--> [" << nodeIndex << "] " << stringifyRole(node.role) // 调试图需要稳定的角色名，便于观察调度前后变化。
           << "\n";
      }
    }
    if (!printedDependency) // 没有边时显式打印 <none>，区分“确实无依赖”和“调试输出漏掉依赖”。
      os << "  <none>\n";
    os << "=== End ScheduleDoubleBufferLoadStoreExt dataflow ===\n";
  }

  std::optional<ExtMove> getLoadMove(Operation *load) const {
    if (load->getBlock() != &block) // 本分析只维护单个 block 内的顺序，跨 block 移动会绕过支配/控制流检查。
      return std::nullopt;

    Operation *anchor = getLoadAnchor(load); // load 尽量提前到所有依赖和前序传输之后，以便更早启动预取。
    if (!anchor || anchor == load || !anchor->isBeforeInBlock(load)) // 插入点没有带来真实位置变化时跳过，避免无意义改写。
      return std::nullopt;

    llvm::SetVector<Operation *> movableOps;
    for (Value operand : load->getOperands())
      if (!collectMovableDefs(operand, &block, anchor, load, movableOps))
        return std::nullopt;

    Value source = getTransferSource(load);
    Value target = getTransferTarget(load);
    if (!source || !target)
      return std::nullopt;

    for (Operation *current = anchor; current && current != load;
         current = current->getNextNode()) {
      if (movableOps.contains(current)) // 这些地址/视图定义会和 load 一起搬走，检查它们等于把同一组移动内部误判成冲突。
        continue;
      if (hasConflictWhenCrossing(current, source, target, aliasAnalysis))
        return std::nullopt;
    }

    ExtMove move;
    move.op = load;
    move.insertionPoint = anchor; // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
    move.movableDefs.assign(movableOps.begin(), movableOps.end()); // 地址/size 计算如果也在移动区间内，必须跟随 load 一起前移。
    return move;
  }

  std::optional<ExtMove> getStoreMove(Operation *store) const {
    if (store->getBlock() != &block) // 本分析只维护单个 block 内的顺序，store 不能被移动到未建模的控制流区域。
      return std::nullopt;

    Operation *insertionPoint = getStoreInsertionPoint(store); // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
    if (!insertionPoint || insertionPoint == store || // 插入点没有带来真实位置变化时跳过，避免无意义改写。
        insertionPoint == store->getNextNode() || // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
        !store->isBeforeInBlock(insertionPoint)) // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
      return std::nullopt;

    Value source = getTransferSource(store);
    Value target = getTransferTarget(store);
    if (!source || !target)
      return std::nullopt;

    for (Operation *current = store->getNextNode(); current &&
         current != insertionPoint; current = current->getNextNode()) { // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
      if (hasConflictWhenCrossing(current, source, target, aliasAnalysis))
        return std::nullopt;
    }

    ExtMove move;
    move.op = store;
    move.insertionPoint = insertionPoint; // 插入点表示该操作在保持依赖后能到达的最早或最晚位置。
    return move;
  }

  void applyMove(const ExtMove &move) const { // 只有分析证明安全后才实际修改 IR 顺序。
    move.op->moveBefore(move.insertionPoint); // 先把 load/store 本身移到插入点，后续 movableDefs 逐个插到它前面，保证操作数在使用之前。
    for (Operation *op : move.movableDefs)
      op->moveBefore(move.op); // 地址/size 定义必须在 load 之前，moveBefore(move.op) 保证每次插入紧挨 load 前方。
  }

private:
  void collectTiles() {
    tileRoots.clear(); // 新的写会截断旧读区间，因此清掉已处理的读者集合。
    for (Operation &op : block.without_terminator()) { // 只分析 loop body 的实际操作，terminator 不参与 tile 数据流。
      Value tile;
      if (auto load = dyn_cast<memref_ext::LoadExOp>(&op)) // load_ex 是输入预取点，它写入本地 tile，需要作为调度边界分析。
        tile = findMemoryRoot(load.getTarget()); // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
      else if (auto load = dyn_cast<memref_ext::BlockDescriptorLoadOp>(&op))
        tile = findMemoryRoot(load.getTarget());
      else if (auto load = dyn_cast<memref_ext::BlockDescLoadOp>(&op))
        tile = findMemoryRoot(load.getTarget());
      else if (auto store = dyn_cast<memref_ext::StoreExOp>(&op)) // store_ex 是输出写回点，它读取本地 tile，需要尽量推迟但不能越过依赖。
        tile = findMemoryRoot(store.getValue()); // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
      else if (auto store = dyn_cast<memref_ext::BlockDescriptorStoreOp>(&op))
        tile = findMemoryRoot(store.getValue());
      else if (auto store = dyn_cast<memref_ext::BlockDescStoreOp>(&op))
        tile = findMemoryRoot(store.getValue());
      else
        continue;

      if (!tile || !tile.getDefiningOp<memref::AllocOp>()) // load 目标必须是本地 alloc tile；外部/参数内存没有明确双缓冲槽，不能纳入重排模型。
        continue;
      if (!llvm::is_contained(tileRoots, tile)) // 同一根 alloc 只分配一个 tileIndex，后续依赖数组才能用索引稳定地合并访问。
        tileRoots.push_back(tile);
    }
  }

  std::optional<unsigned> getTileIndex(Value value) const {
    value = findMemoryRoot(value); // 把不同视图规约到同一根值，别名和 tile 匹配才不会漏判。
    if (!value) // 追不到根值时无法和 tileRoots 做别名比较，返回空索引让调用方按保守路径处理。
      return std::nullopt;
    for (auto [index, tile] : llvm::enumerate(tileRoots))
      if (value == tile || aliasAnalysis.alias(value, tile)) // 值等于或可能别名某个 tile root 时，依赖分析必须把它归到同一 tileIndex。
        return index;
    return std::nullopt;
  }

  std::optional<unsigned> getNodeIndex(Operation *op) const {
    auto it = opToNode.find(op);
    if (it == opToNode.end()) // op 没有 tile 访问摘要时不在调度图中，不能计算它相对依赖边的合法位置。
      return std::nullopt;
    return it->second;
  }

  std::optional<ExtNode> collectNode(Operation *op) const {
    ExtNode node;
    node.op = op;

    if (auto load = dyn_cast<memref_ext::LoadExOp>(op)) { // load_ex 是输入预取点，它写入本地 tile，需要作为调度边界分析。
      node.role = ExtScheduleRole::Load;
      if (auto target = getTileIndex(load.getTarget())) // 只有目标能映射到已知 tile，load_ex 才会改变这个 tile 的版本并产生写依赖。
        addAccess(node.accesses, *target, /*reads=*/false, /*writes=*/true);
      return node;
    }

    if (auto load = dyn_cast<memref_ext::BlockDescriptorLoadOp>(op)) {
      node.role = ExtScheduleRole::Load;
      if (auto target = getTileIndex(load.getTarget()))
        addAccess(node.accesses, *target, /*reads=*/false, /*writes=*/true);
      return node;
    }

    if (auto load = dyn_cast<memref_ext::BlockDescLoadOp>(op)) {
      node.role = ExtScheduleRole::Load;
      if (auto target = getTileIndex(load.getTarget()))
        addAccess(node.accesses, *target, /*reads=*/false, /*writes=*/true);
      return node;
    }

    if (auto store = dyn_cast<memref_ext::StoreExOp>(op)) { // store_ex 是输出写回点，它读取本地 tile，需要尽量推迟但不能越过依赖。
      node.role = ExtScheduleRole::Store;
      if (auto source = getTileIndex(store.getValue())) // 只有源能映射到已知 tile，store_ex 才消费该 tile 当前内容并产生读依赖。
        addAccess(node.accesses, *source, /*reads=*/true, /*writes=*/false);
      return node;
    }

    if (auto store = dyn_cast<memref_ext::BlockDescriptorStoreOp>(op)) {
      node.role = ExtScheduleRole::Store;
      if (auto source = getTileIndex(store.getValue()))
        addAccess(node.accesses, *source, /*reads=*/true, /*writes=*/false);
      return node;
    }

    if (auto store = dyn_cast<memref_ext::BlockDescStoreOp>(op)) {
      node.role = ExtScheduleRole::Store;
      if (auto source = getTileIndex(store.getValue()))
        addAccess(node.accesses, *source, /*reads=*/true, /*writes=*/false);
      return node;
    }

    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp) // 不能枚举副作用时无法判断它是否访问 tile；这里不生成精确节点，避免构造错误依赖。
      return std::nullopt;

    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects); // 读取精确副作用列表，尽量少把无关操作当成移动屏障。
    for (const MemoryEffects::EffectInstance &effect : effects) {
      bool reads = isa<MemoryEffects::Read>(effect.getEffect());
      bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (!reads && !writes) // 非读写副作用不改变 tile 数据流，不需要加入节点访问摘要。
        continue;

      Value value = effect.getValue(); // 副作用必须有具体内存值，否则无法证明与 tile 无关。
      if (!value) { // effect 没有关联 SSA memref 时无法做别名过滤，只能把它升级成未知内存屏障。
        node.hasUnknownMemoryEffect = true; // 未知副作用要当成访问所有 tile，避免错误重排。
        continue;
      }
      if (auto tileIndex = getTileIndex(value)) // effect 关联的 memref 命中 tileRoots 时，才需要把该读写纳入调度依赖。
        addAccess(node.accesses, *tileIndex, reads, writes);
    }

    if (node.accesses.empty() && !node.hasUnknownMemoryEffect) // 既没有已知 tile 访问也没有未知内存屏障时，该 op 对调度模型不可见。
      return std::nullopt;
    return node;
  }

  SmallVector<ExtAccess> getDependencyAccesses(const ExtNode &node) const {
    SmallVector<ExtAccess> accesses(node.accesses);
    if (!node.hasUnknownMemoryEffect) // 节点只有已知 tile 访问时直接使用精确集合，避免无谓地阻止 load/store 移动。
      return accesses; // 把未知副作用展开后的访问集合交给依赖构建使用。
    for (auto [tileIndex, tile] : llvm::enumerate(tileRoots))
      addAccess(accesses, tileIndex, /*reads=*/true, /*writes=*/true);
    return accesses; // 把未知副作用展开后的访问集合交给依赖构建使用。
  }

  void buildDependencies() {
    SmallVector<std::optional<unsigned>> lastWrite(tileRoots.size());
    SmallVector<SmallVector<unsigned>> readsSinceWrite(tileRoots.size());

    for (auto [nodeIndex, node] : llvm::enumerate(nodes)) {
      SmallVector<ExtEdge> edges;
      for (const ExtAccess &access : getDependencyAccesses(node)) { // 逐个 tile 读写关系转成 RAW/WAR/WAW 依赖。
        bool reads = access.reads;
        bool writes = access.writes;
        unsigned tileIndex = access.tileIndex;

        if (reads && lastWrite[tileIndex]) // 当前读会观察最近写出的 tile 版本，因此要从最近写者连 RAW 边。
          addOrMergeEdge(edges, *lastWrite[tileIndex], tileIndex,
                         /*raw=*/true, /*war=*/false, /*waw=*/false); // RAW 表示当前读依赖前序写，load/store 移动不能破坏它。

        if (writes) { // 写操作会结束此前读者的开放区间，并成为新的最近写。
          if (lastWrite[tileIndex]) // 同一 tile 已有前序写时，当前写必须连 WAW 边以防最终值来源被交换。
            addOrMergeEdge(edges, *lastWrite[tileIndex], tileIndex,
                           /*raw=*/false, /*war=*/false, /*waw=*/true); // RAW 表示当前读依赖前序写，load/store 移动不能破坏它。
          for (unsigned read : readsSinceWrite[tileIndex])
            addOrMergeEdge(edges, read, tileIndex, /*raw=*/false,
                           /*war=*/true, /*waw=*/false); // WAR 表示后续写不能越过此前读，防止读到新值。
          readsSinceWrite[tileIndex].clear();
          lastWrite[tileIndex] = nodeIndex;
          if (reads) // 读写同一节点在写后仍算读者，后续写不能越过它破坏其读到的旧值。
            readsSinceWrite[tileIndex].push_back(nodeIndex);
        } else if (reads) {
          readsSinceWrite[tileIndex].push_back(nodeIndex);
        }
      }
      llvm::sort(edges, [](const ExtEdge &lhs, const ExtEdge &rhs) { // 稳定排序让调试输出和重写决策不依赖哈希/集合顺序。
        if (lhs.predecessor != rhs.predecessor) // 不同前驱先按 IR 节点顺序排序，保证依赖图打印和测试结果可复现。
          return lhs.predecessor < rhs.predecessor;
        return lhs.tileIndex < rhs.tileIndex;
      });
      node.predecessors = std::move(edges);
    }
  }

  void addOrMergeEdge(SmallVectorImpl<ExtEdge> &edges, unsigned predecessor,
                      unsigned tileIndex, bool raw, bool war, bool waw) const {
    for (ExtEdge &edge : edges) {
      if (edge.predecessor == predecessor && edge.tileIndex == tileIndex) { // 同一前驱对同一 tile 可能同时形成 RAW/WAR/WAW，合并可保留完整约束且避免重复边。
        edge.raw |= raw; // RAW 表示当前读依赖前序写，load/store 移动不能破坏它。
        edge.war |= war; // WAR 表示后续写不能越过此前读，防止读到新值。
        edge.waw |= waw; // WAW 表示多个写必须保持最终覆盖顺序。
        return;
      }
    }
    edges.push_back(ExtEdge{predecessor, tileIndex, raw, war, waw});
  }

  Operation *getLoadAnchor(Operation *op) const { // load 尽量提前到所有依赖和前序传输之后，以便更早启动预取。
    std::optional<unsigned> nodeIndex = getNodeIndex(op);
    if (!nodeIndex) // load 未进入依赖图说明缺少 tile 访问信息，不能推导它能前移到哪里。
      return nullptr;

    std::optional<unsigned> latestBarrier; // 记录 load 不能越过的最晚屏障，最终插到屏障之后。
    for (const ExtEdge &edge : nodes[*nodeIndex].predecessors)
      if (!latestBarrier || edge.predecessor > *latestBarrier) // load 不能越过任何前驱依赖，因此锚点要取所有前驱中最靠后的一个。
        latestBarrier = edge.predecessor;

    for (unsigned index = 0; index < *nodeIndex; ++index) {
      if (!isCopyLikeRole(nodes[index].role)) // 普通 compute 已由数据依赖约束；这里额外只给 load/store 传输类节点保序。
        continue;
      if (!latestBarrier || index > *latestBarrier) // 前序传输不能被新的 load 越过，取最靠后的传输作为额外屏障。
        latestBarrier = index; // 记录 load 不能越过的最晚屏障，最终插到屏障之后。
    }

    if (!latestBarrier) // 没有数据依赖和传输保序屏障时，最早合法位置就是 block 的第一个操作前。
      return &block.front();
    return nodes[*latestBarrier].op->getNextNode();
  }

  Operation *getStoreInsertionPoint(Operation *op) const { // store 尽量后移到所有消费者之后，但不能越过后续传输或依赖它的节点。
    std::optional<unsigned> nodeIndex = getNodeIndex(op);
    if (!nodeIndex) // store 未进入依赖图说明缺少 tile 访问信息，不能推导它能后移到哪里。
      return nullptr;

    std::optional<unsigned> earliestBarrier; // 记录 store 不能越过的最早后继屏障，最终插到它之前。
    for (auto [successorIndex, node] : llvm::enumerate(nodes)) {
      if (successorIndex <= *nodeIndex) // 后移 store 只可能跨过原位置之后的节点，原位置之前的依赖已经满足。
        continue;
      if (isCopyLikeRole(node.role)) { // 后续 load/store_ext 属于传输序列边界，store 后移不能改变它们之间的 DMA 发起顺序。
        if (!earliestBarrier || successorIndex < *earliestBarrier) // 多个后续传输中最早那个最先限制 store，插入点必须在它之前。
          earliestBarrier = successorIndex; // 记录 store 不能越过的最早后继屏障，最终插到它之前。
        continue;
      }
      for (const ExtEdge &edge : node.predecessors) {
        if (edge.predecessor != *nodeIndex) // 只有以后当前 store 为前驱的边才说明该后继会观察 store 的原始顺序。
          continue;
        if (!earliestBarrier || successorIndex < *earliestBarrier) // 第一个依赖当前 store 的后继就是数据流消费者，store 不能跨过去。
          earliestBarrier = successorIndex; // 记录 store 不能越过的最早后继屏障，最终插到它之前。
      }
    }

    if (!earliestBarrier) // 没有后续传输或消费者限制时，store 的最晚合法位置是 terminator 之前。
      return block.getTerminator();
    return nodes[*earliestBarrier].op;
  }

  Block &block;
  AliasAnalysis &aliasAnalysis;
  SmallVector<Value> tileRoots;
  SmallVector<ExtNode> nodes;
  llvm::DenseMap<Operation *, unsigned> opToNode;
};

bool scheduleBlock(Block &block, AliasAnalysis &aliasAnalysis) { // 在一个 loop body 内反复调整 load/store，直到局部顺序稳定。
  if (!block.getTerminator()) // store 后移需要 terminator 作为兜底插入点；没有 terminator 时放弃整个 block 的调度。
    return false;

  LoadStoreExtDataFlowAnalysis before(block, aliasAnalysis);
  before.build();
  before.print(llvm::outs(), "BEFORE"); // 打印调度前图，便于确认依赖是否限制了预期移动。

  bool changed = false;
  bool localChanged = true; // 一次移动会改变后续锚点，所以需要继续迭代到不再变化。
  while (localChanged) { // 反复重建依赖图并移动，直到本轮不再产生变化。
    localChanged = false; // 一次移动会改变后续锚点，所以需要继续迭代到不再变化。

    LoadStoreExtDataFlowAnalysis analysis(block, aliasAnalysis);
    analysis.build(); // 每轮移动后重建依赖图，避免用过期节点顺序继续决策。

    for (Operation *load : analysis.getLoads()) { // 按当前 IR 顺序尝试前移每个 load_ex。
      if (std::optional<ExtMove> move = analysis.getLoadMove(load)) { // 返回 move 说明锚点、操作数支配和跨越冲突都已通过，才允许改写 IR 顺序。
        analysis.applyMove(*move); // 只有分析证明安全后才实际修改 IR 顺序。
        changed = true;
        localChanged = true; // 一次移动会改变后续锚点，所以需要继续迭代到不再变化。
      }
    }

    analysis.build(); // 每轮移动后重建依赖图，避免用过期节点顺序继续决策。

    for (Operation *store : llvm::reverse(analysis.getStores())) { // 从后往前尝试后移 store_ex，减少互相干扰。
      if (std::optional<ExtMove> move = analysis.getStoreMove(store)) { // 返回 move 说明后移插入点和跨越冲突都已验证，才允许改写 IR 顺序。
        analysis.applyMove(*move); // 只有分析证明安全后才实际修改 IR 顺序。
        changed = true;
        localChanged = true; // 一次移动会改变后续锚点，所以需要继续迭代到不再变化。
      }
    }
  }

  LoadStoreExtDataFlowAnalysis after(block, aliasAnalysis);
  after.build();
  after.print(llvm::outs(), "AFTER"); // 打印调度后图，便于对比 load/store 是否被合法重排。

  return changed; // 告诉 pass manager 本 block 的操作顺序是否被改写。
}

struct ScheduleDoubleBufferLoadStoreExtPass
    : public ::impl::ScheduleDoubleBufferLoadStoreExtBase<
          ScheduleDoubleBufferLoadStoreExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override { // 注册这些方言，保证 pass 创建或解析相关 op 时 dialect 已加载。
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    memref_ext::MemRefExtDialect, scf::SCFDialect,
                    triton::TritonDialect>();
  }

  void runOnOperation() override {
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>(); // 复用 MLIR 的别名分析来减少保守屏障。

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kLoopKindAttr)) // 只有前序 pass 标记过 loop kind 的循环才符合 double-buffer 调度假设，普通循环保持不动。
        loops.push_back(loop);
    });

    for (scf::ForOp loop : loops)
      scheduleBlock(*loop.getBody(), aliasAnalysis); // 在一个 loop body 内反复调整 load/store，直到局部顺序稳定。
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createScheduleDoubleBufferLoadStoreExtPass() {
  return std::make_unique<ScheduleDoubleBufferLoadStoreExtPass>(); // 把 pass 实例交给 MLIR pass pipeline 管理生命周期。
}

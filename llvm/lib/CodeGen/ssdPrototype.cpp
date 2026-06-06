//===-- SimilarSequenceDetector.cpp - Differential Outlining Detection ----===//
//
// Algorithm:
//   For each MachineBasicBlock ("anchor"):
//     1. Build a suffix array over its opcode sequence.
//     2. Build the LCP array (Kasai's algorithm).
//     3. Walk LCP intervals to find all repeated substrings of length >=
//        MinSequenceLen within the anchor block.
//     4. For each such substring (unique by opcode sequence), scan every
//        other MBB in the module for occurrences of the same opcode sequence
//        at any position (substring match, not just prefix).
//     5. Group all occurrences (anchor + matches) into a CandidateGroup.
//        Record operand diffs for each occurrence relative to the first.
//
// Output: printed to errs(). No IR/MIR is modified.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/Module.h"
#include "llvm/PassRegistry.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace llvm {
ModulePass *createSimilarSequenceDetectorPass();
void initializeSimilarSequenceDetectorPass(PassRegistry &);
} // namespace llvm

using namespace llvm;

static constexpr unsigned MinSequenceLen = 4;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// A flat list of instructions from one MBB.
using InstrVec = SmallVector<const MachineInstr *, 32>;

// One operand slot difference relative to the reference occurrence.
struct OperandDiff {
  unsigned InstrOffset;  // Offset within the substring.
  unsigned OperandIndex;
};

// One occurrence of a candidate substring within some MBB.
struct Occurrence {
  const MachineBasicBlock *MBB;
  unsigned StartIndex;   // Offset into that MBB's InstrVec.
  SmallVector<OperandDiff, 8> Diffs; // vs. reference (first occurrence).
};

// A group of all occurrences of one repeated opcode sequence.
struct CandidateGroup {
  unsigned Len;                        // Substring length in instructions.
  SmallVector<Occurrence, 8> Occurrences;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool shouldSkip(const MachineInstr &MI) {
  return MI.isDebugInstr() || MI.isCFIInstruction() || MI.isLabel();
}

static InstrVec collectInstrs(const MachineBasicBlock &MBB) {
  InstrVec R;
  for (const MachineInstr &MI : MBB)
    if (!shouldSkip(MI))
      R.push_back(&MI);
  return R;
}

static bool operandsEqual(const MachineOperand &A, const MachineOperand &B) {
  if (A.getType() != B.getType()) return false;
  switch (A.getType()) {
  case MachineOperand::MO_Register:      return A.getReg() == B.getReg();
  case MachineOperand::MO_Immediate:     return A.getImm() == B.getImm();
  case MachineOperand::MO_FPImmediate:   return A.getFPImm() == B.getFPImm();
  case MachineOperand::MO_MachineBasicBlock: return A.getMBB() == B.getMBB();
  case MachineOperand::MO_GlobalAddress:
    return A.getGlobal() == B.getGlobal() && A.getOffset() == B.getOffset();
  case MachineOperand::MO_ExternalSymbol:
    return StringRef(A.getSymbolName()) == StringRef(B.getSymbolName());
  default: return true;
  }
}

// Compute operand diffs between a reference substring and a candidate
// substring starting at CandStart in CandInstrs, length Len.
static SmallVector<OperandDiff, 8>
computeDiffs(const InstrVec &RefInstrs, unsigned RefStart,
             const InstrVec &CandInstrs, unsigned CandStart,
             unsigned Len) {
  SmallVector<OperandDiff, 8> Diffs;
  for (unsigned I = 0; I < Len; ++I) {
    const MachineInstr &R = *RefInstrs[RefStart + I];
    const MachineInstr &C = *CandInstrs[CandStart + I];
    unsigned NOps = std::min(R.getNumOperands(), C.getNumOperands());
    for (unsigned J = 0; J < NOps; ++J)
      if (!operandsEqual(R.getOperand(J), C.getOperand(J)))
        Diffs.push_back({I, J});
    unsigned MaxOps = std::max(R.getNumOperands(), C.getNumOperands());
    for (unsigned J = NOps; J < MaxOps; ++J)
      Diffs.push_back({I, J});
  }
  return Diffs;
}

//===----------------------------------------------------------------------===//
// Suffix Array + LCP (over opcode sequences)
//===----------------------------------------------------------------------===//

// Build suffix array for opcodes[0..n-1] using O(n log n) prefix doubling.
static std::vector<int> buildSuffixArray(const std::vector<unsigned> &S) {
  int n = (int)S.size();
  if (n == 0) return {};

  std::vector<int> SA(n), Rank(n), Tmp(n);
  for (int i = 0; i < n; ++i) { SA[i] = i; Rank[i] = (int)S[i]; }

  for (int Gap = 1; Gap < n; Gap <<= 1) {
    auto cmp = [&](int a, int b) {
      if (Rank[a] != Rank[b]) return Rank[a] < Rank[b];
      int ra = a + Gap < n ? Rank[a + Gap] : -1;
      int rb = b + Gap < n ? Rank[b + Gap] : -1;
      return ra < rb;
    };
    std::sort(SA.begin(), SA.end(), cmp);
    Tmp[SA[0]] = 0;
    for (int i = 1; i < n; ++i)
      Tmp[SA[i]] = Tmp[SA[i-1]] + (cmp(SA[i-1], SA[i]) ? 1 : 0);
    Rank = Tmp;
    if (Rank[SA[n-1]] == n-1) break;
  }
  return SA;
}

// Kasai's algorithm: build LCP array from SA in O(n).
static std::vector<int> buildLCP(const std::vector<unsigned> &S,
                                  const std::vector<int> &SA) {
  int n = (int)S.size();
  std::vector<int> Rank(n), LCP(n, 0);
  for (int i = 0; i < n; ++i) Rank[SA[i]] = i;
  int H = 0;
  for (int i = 0; i < n; ++i) {
    if (Rank[i] > 0) {
      int j = SA[Rank[i] - 1];
      while (i + H < n && j + H < n && S[i+H] == S[j+H]) ++H;
      LCP[Rank[i]] = H;
      if (H > 0) --H;
    }
  }
  return LCP;
}

// Extract all maximal LCP intervals with LCP value >= MinLen.
// Returns (lcp_value, [SA positions]) — each SA position is a suffix start.
static void extractIntervals(
    const std::vector<int> &SA,
    const std::vector<int> &LCP,
    unsigned MinLen,
    SmallVectorImpl<std::pair<unsigned, std::vector<int>>> &Out) {

  int n = (int)SA.size();
  // Stack-based LCP interval enumeration.
  struct Frame { int lcp; int start; };
  std::vector<Frame> Stack;
  Stack.push_back({0, 0});

  for (int i = 1; i <= n; ++i) {
    int curLcp = (i < n) ? LCP[i] : 0;
    int start = i - 1;
    while (!Stack.empty() && Stack.back().lcp > curLcp) {
      auto [lcp, s] = Stack.back(); Stack.pop_back();
      if ((unsigned)lcp >= MinLen) {
        std::vector<int> Positions;
        for (int j = s; j < i; ++j) Positions.push_back(SA[j]);
        Out.push_back({(unsigned)lcp, std::move(Positions)});
      }
      start = s;
    }
    if (Stack.empty() || Stack.back().lcp < curLcp)
      Stack.push_back({curLcp, start});
  }
}

//===----------------------------------------------------------------------===//
// Scan one MBB for an opcode sequence match at any position
//===----------------------------------------------------------------------===//

// Returns all start positions in Haystack where the opcode sequence
// Needle[0..Len-1] appears.
static SmallVector<unsigned, 4>
findOccurrences(const std::vector<unsigned> &Needle, unsigned Len,
                const std::vector<unsigned> &Haystack) {
  SmallVector<unsigned, 4> Hits;
  if (Haystack.size() < Len) return Hits;
  for (unsigned I = 0; I + Len <= Haystack.size(); ++I) {
    bool Match = true;
    for (unsigned J = 0; J < Len; ++J)
      if (Haystack[I+J] != Needle[J]) { Match = false; break; }
    if (Match) Hits.push_back(I);
  }
  return Hits;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SimilarSequenceDetector : public ModulePass {
  static char ID;
  SimilarSequenceDetector() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "Similar Sequence Detector (Differential Outlining Prototype)";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  bool runOnModule(Module &M) override {
    MachineModuleInfo &MMI =
        getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

    // --- Collect all MBBs ---
    using MBBData = std::pair<const MachineBasicBlock *, InstrVec>;
    SmallVector<MBBData, 256> AllMBBs;

    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      MachineFunction *MF = MMI.getMachineFunction(F);
      if (!MF) continue;
      for (const MachineBasicBlock &MBB : *MF) {
        auto IV = collectInstrs(MBB);
        if (IV.size() < MinSequenceLen) continue;
        AllMBBs.push_back({&MBB, std::move(IV)});
      }
    }

    if (AllMBBs.empty()) {
      errs() << "[SSD] No qualifying MBBs.\n";
      return false;
    }

    // Precompute opcode vectors for every MBB.
    SmallVector<std::vector<unsigned>, 256> OpcodeVecs(AllMBBs.size());
    for (unsigned I = 0; I < AllMBBs.size(); ++I)
      for (const MachineInstr *MI : AllMBBs[I].second)
        OpcodeVecs[I].push_back((unsigned)MI->getOpcode());

    // Key = opcode sequence (to deduplicate groups across anchors).
    std::map<std::vector<unsigned>, bool> Seen;

    SmallVector<CandidateGroup, 64> Groups;

    // --- For each anchor MBB, find repeated substrings via SA+LCP ---
    for (unsigned AnchorIdx = 0; AnchorIdx < AllMBBs.size(); ++AnchorIdx) {
      const auto &[AnchorMBB, AnchorIV] = AllMBBs[AnchorIdx];
      const std::vector<unsigned> &AnchorOps = OpcodeVecs[AnchorIdx];

      if (AnchorOps.size() < MinSequenceLen) continue;

      auto SA  = buildSuffixArray(AnchorOps);
      auto LCP = buildLCP(AnchorOps, SA);

      SmallVector<std::pair<unsigned, std::vector<int>>, 16> Intervals;
      extractIntervals(SA, LCP, MinSequenceLen, Intervals);

      for (auto &[SubLen, Positions] : Intervals) {
        // The canonical opcode sequence for this interval.
        unsigned RefPos = (unsigned)Positions[0];
        if (RefPos + SubLen > AnchorOps.size()) continue;

        std::vector<unsigned> Key(AnchorOps.begin() + RefPos,
                                  AnchorOps.begin() + RefPos + SubLen);
        if (Seen.count(Key)) continue;
        Seen[Key] = true;

        // --- Scan all MBBs for this opcode sequence ---
        CandidateGroup G;
        G.Len = SubLen;

        // Reference occurrence: first position in anchor.
        Occurrence Ref;
        Ref.MBB = AnchorMBB;
        Ref.StartIndex = RefPos;
        // No diffs vs itself.
        G.Occurrences.push_back(std::move(Ref));

        for (unsigned MIdx = 0; MIdx < AllMBBs.size(); ++MIdx) {
          const auto &[MMBB, MIV] = AllMBBs[MIdx];
          const std::vector<unsigned> &MOps = OpcodeVecs[MIdx];

          auto Hits = findOccurrences(Key, SubLen, MOps);
          for (unsigned HitPos : Hits) {
            // Skip the reference occurrence itself.
            if (MMBB == AnchorMBB && HitPos == RefPos) continue;

            Occurrence Occ;
            Occ.MBB = MMBB;
            Occ.StartIndex = HitPos;
            Occ.Diffs = computeDiffs(AnchorIV, RefPos, MIV, HitPos, SubLen);
            G.Occurrences.push_back(std::move(Occ));
          }
        }

        if (G.Occurrences.size() >= 2)
          Groups.push_back(std::move(G));
      }
    }

    // --- Maximality filter ---
    // Remove group G if every one of its occurrences (MBB, StartIndex) is
    // also covered by a strictly longer group G2 whose occurrence starts at
    // the same position. Such a G is a strict prefix of G2 and adds no
    // information beyond what G2 already reports.
    //
    // Build a set of (MBB, start) pairs covered by each group for fast lookup.
    using OccKey = std::pair<const MachineBasicBlock *, unsigned>;
    std::vector<std::set<OccKey>> OccSets(Groups.size());
    for (unsigned I = 0; I < Groups.size(); ++I)
      for (const Occurrence &O : Groups[I].Occurrences)
        OccSets[I].insert({O.MBB, O.StartIndex});

    std::vector<bool> Suppress(Groups.size(), false);
    for (unsigned I = 0; I < Groups.size(); ++I) {
      if (Suppress[I]) continue;
      for (unsigned J = 0; J < Groups.size(); ++J) {
        if (I == J || Suppress[J]) continue;
        if (Groups[J].Len <= Groups[I].Len) continue;
        // Check if every occurrence of I is present in J at the same position.
        bool AllCovered = true;
        for (const OccKey &K : OccSets[I]) {
          if (!OccSets[J].count(K)) { AllCovered = false; break; }
        }
        if (AllCovered) { Suppress[I] = true; break; }
      }
    }

    // --- Print results ---
    unsigned ExactGroups = 0;
    unsigned PrintedGroups = 0;
    for (unsigned GI = 0; GI < Groups.size(); ++GI) {
      if (Suppress[GI]) continue;
      const CandidateGroup &G = Groups[GI];
      ++PrintedGroups;
      bool Exact = true;
      for (unsigned I = 1; I < G.Occurrences.size(); ++I)
        if (!G.Occurrences[I].Diffs.empty()) { Exact = false; break; }
      if (Exact) ++ExactGroups;

      errs() << "=== Candidate Group | len=" << G.Len
             << " | occurrences=" << G.Occurrences.size()
             << (Exact ? " | EXACT" : " | DIFFERENTIAL") << " ===\n";

#ifndef NO_PRINT_CANDIDATE_GROUP
      for (unsigned I = 0; I < G.Occurrences.size(); ++I) {
        const Occurrence &Occ = G.Occurrences[I];
        errs() << "  [" << I << "] "
               << Occ.MBB->getParent()->getName() << "::"
               << Occ.MBB->getName()
               << " @ instr " << Occ.StartIndex;
        if (I == 0) {
          errs() << " [reference]\n";
        } else if (Occ.Diffs.empty()) {
          errs() << " [exact]\n";
        } else {
          errs() << " diffs=" << Occ.Diffs.size() << " : ";
          unsigned PrevOff = UINT_MAX;
          for (const OperandDiff &D : Occ.Diffs) {
            if (D.InstrOffset != PrevOff) {
              errs() << "instr+" << D.InstrOffset << ":{";
              PrevOff = D.InstrOffset;
            }
            errs() << D.OperandIndex << " ";
          }
          errs() << "}\n";
        }
      }
      errs() << "\n";
#endif // NO_PRINT_CANDIDATE_GROUP
    }

    errs() << "[SSD] " << PrintedGroups << " maximal groups ("
           << ExactGroups << " exact, "
           << (PrintedGroups - ExactGroups) << " differential).\n";

    return false;
  }
};

char SimilarSequenceDetector::ID = 0;

INITIALIZE_PASS_BEGIN(SimilarSequenceDetector, "similar-sequence-detector",
                      "Detect similar MBB sequences for differential outlining",
                      false, true)
INITIALIZE_PASS_DEPENDENCY(MachineModuleInfoWrapperPass)
INITIALIZE_PASS_END(SimilarSequenceDetector, "similar-sequence-detector",
                    "Detect similar MBB sequences for differential outlining",
                    false, true)

ModulePass *llvm::createSimilarSequenceDetectorPass() {
  return new SimilarSequenceDetector();
}

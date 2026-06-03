#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "similar-sequence-detector"

namespace {
class SimilarSequenceDetector : public MachineFunctionPass {
public:
  static char ID;

  SimilarSequenceDetector() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Similar Sequence Detector";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    errs() << "SSD saw function: " << MF.getName() << "\n";
    return false;
  }
};
}

char SimilarSequenceDetector::ID = 0;

INITIALIZE_PASS(SimilarSequenceDetector,
                DEBUG_TYPE,
                "Detect similar machine instruction sequences",
                false,
                true)

namespace llvm {
FunctionPass *createSimilarSequenceDetectorPass() {
  return new SimilarSequenceDetector();
}
} // namespace llvm

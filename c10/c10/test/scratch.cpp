#include <c10/c10.h>

#include <c10/cpu/CPUTensorImpl.h>

using namespace c10;

// Just a scratch pad for running little programs
int main() {
  Tensor x = c10::cpu::CPUTensorImpl::HACK_tensor(ScalarType::Int, {3, 2}, {2, 1});
  x.resize_({5, 5}, {5, 1});
}
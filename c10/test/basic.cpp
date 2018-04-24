#include <c10.h>

using namespace c10;

// Just a scratch pad for running little programs
int main() {
  Tensor x = empty({3, 2}, float32);
  x.legacy_pytorch_resize_({5, 5});
  Tensor y = tensor<std::string>({"foo", "bar"}, {1,2});
}

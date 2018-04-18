#include <c10.h>

using namespace c10;

// Just a scratch pad for running little programs
int main() {
  Tensor x = tensor(float32, {3, 2}, {2, 1});
  x.resize_({5, 5}, {5, 1});

  Tensor y = tensor<std::string>({1,2}, {"foo", "bar"});
}

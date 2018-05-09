#include <c10/dispatch/OpSig.h>

#include <iostream>

using namespace c10;

constexpr auto equals() {
  return OpSig<0>()
      .arg<int>("lhs")
      .arg<int>("rhs");
}

int main() {
  std::cerr << equals().args[0].name << "\n";
  std::cerr << equals().args[1].name << "\n";
  return 0;
}

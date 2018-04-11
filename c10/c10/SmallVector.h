#pragma once

#include <vector>

// TODO: Fill in an actual SmallVector implementation here.  Both Folly and LLVM's
// implementation are a bit annoying to make standalone.  Maybe this can be made
// simpler by assuming T is POD.
// TODO: For the common case of sizes and strides, the lengths of the two arrays
// are equal, so there is no need to store the ndim twice.  Worth thinking about.

namespace c10 {
template <typename T>
using SmallVector = std::vector<T>;

}

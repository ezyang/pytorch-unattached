#include <gtest/gtest.h>

#include <c10/op/All.h>
#include <c10.h>

using namespace c10;

/*
TEST(All_test, tensor) {
  DimVector sizes = {2,3};
  DimVector strides = {3,1};
  auto r = op::tensor(int32, sizes, strides);
  ASSERT_EQ(r.dtype(), int32);
  ASSERT_TRUE(r.sizes().equals(sizes));
  ASSERT_TRUE(r.strides().equals(strides));
}
 */

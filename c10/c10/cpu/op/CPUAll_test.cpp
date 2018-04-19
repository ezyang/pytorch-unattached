#include <gtest/gtest.h>
#include <c10/cpu/op/CPUAll.h>
#include <c10.h>

using namespace c10;

using I = ArrayRef<int64_t>;
static auto shapes = ::testing::Values(I{}, I{0}, I{1}, I{0,1}, I{1,1}, I{2,1}, I{2,2}, I{2,2,2});

class CPUAll_empty_test : public ::testing::TestWithParam<ArrayRef<int64_t>> {};
TEST_P(CPUAll_empty_test, int32) {
  ArrayRef<int64_t> size = GetParam();
  constexpr DataType dtype = int32;

  Tensor x = cpu::op::empty(size, dtype);
  ASSERT_EQ(x.dtype(), dtype);
  ASSERT_TRUE(x.sizes().equals(size));
}
INSTANTIATE_TEST_CASE_P(CPUAll_empty_test_shapes, CPUAll_empty_test, shapes);

class CPUAll_zeros_test : public ::testing::TestWithParam<ArrayRef<int64_t>> {};
TEST_P(CPUAll_zeros_test, int32) {
  ArrayRef<int64_t> size = GetParam();
  using dtype_t = int32_t;
  constexpr DataType dtype = c10::dtype<dtype_t>();

  Tensor x = cpu::op::zeros(size, dtype);
  ASSERT_EQ(x.dtype(), dtype);
  ASSERT_EQ(x.numel(), product(size));
  ASSERT_TRUE(x.sizes().equals(size));
  auto* p = x.data<dtype_t>();
  for (int i = 0; i < x.numel(); i++) {
    ASSERT_EQ(p[i], 0);
  }
}
INSTANTIATE_TEST_CASE_P(CPUAll_zeros_test_shapes, CPUAll_zeros_test, shapes);

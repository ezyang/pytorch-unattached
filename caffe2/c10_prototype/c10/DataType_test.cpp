#include <gtest/gtest.h>
#include <c10/DataType.h>

using namespace c10;

TEST(DataType_test, int32) {
  EXPECT_EQ(int32, dtype<int32_t>());
  EXPECT_EQ(int32.itemsize(), 4);
  EXPECT_EQ(int32.ctor(), nullptr);
  EXPECT_EQ(int32.copy(), nullptr);
  EXPECT_EQ(int32.dtor(), nullptr);
  EXPECT_STREQ(int32.name(), "int32");
}


TEST(DataType_test, string) {
  EXPECT_EQ(string_dtype, dtype<std::string>());
  EXPECT_EQ(string_dtype.itemsize(), sizeof(std::string));
  EXPECT_STREQ(string_dtype.name(), "string");
  void *p = std::malloc(sizeof(std::string) * 3);
  string_dtype.ctor()(p, 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(static_cast<std::string *>(p)[i], std::string(""));
  }
  std::vector<std::string> x{"one", "two", "three"};
  string_dtype.copy()(x.data(), p, 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(static_cast<std::string *>(p)[i], x[i]);
  }
  string_dtype.dtor()(p, 3);
  std::free(p);
}

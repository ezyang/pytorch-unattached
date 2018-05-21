#include "Metaprogramming.h"
#include <gtest/gtest.h>

using namespace c10::guts;

namespace {

namespace test_eq {
    static_assert(eq(std::array < int, 3 > {{2, 3, 4}}, std::array < int, 3 > {{2, 3, 4}}), "");
    static_assert(!eq(std::array < int, 3 > {{2, 3, 4}}, std::array < int, 3 > {{2, 5, 4}}), "");
}

namespace test_tail {
    static_assert(eq(std::array < int, 2 > {{3, 4}}, tail(std::array < int, 3 > {{2, 3, 4}})), "");
    static_assert(eq(std::array < int, 0 > {{}}, tail(std::array < int, 1 > {{3}})), "");
}

namespace test_prepend {
    static_assert(eq(std::array < int, 3 > {{2, 3, 4}}, prepend(2, std::array < int, 2 > {{3, 4}})), "");
    static_assert(eq(std::array < int, 1 > {{3}}, prepend(3, std::array < int, 0 > {{}})), "");
}

namespace test_function_traits {
    static_assert(std::is_same<void, typename function_traits<void(int, float)>::return_type>::value, "");
    static_assert(std::is_same<int, typename function_traits<int(int, float)>::return_type>::value, "");
    static_assert(std::is_same<typelist::typelist<int, float>, typename function_traits<void(int, float)>::parameter_types>::value, "");
    static_assert(std::is_same<typelist::typelist<int, float>, typename function_traits<int(int, float)>::parameter_types>::value, "");
}

namespace test_to_std_array {
    constexpr int obj2[3] = {3, 5, 6};
    static_assert(eq(std::array < int, 3 > {{3, 5, 6}}, to_std_array(obj2)), "");
    static_assert(eq(std::array < int, 3 > {{3, 5, 6}}, to_std_array({3, 5, 6})), "");
}

struct MovableOnly {
    constexpr MovableOnly(int val_): val(val_) {/* no default constructor */}
    MovableOnly(const MovableOnly&) = delete;
    MovableOnly(MovableOnly&&) = default;
    MovableOnly& operator=(const MovableOnly&) = delete;
    MovableOnly& operator=(MovableOnly&&) = default;

    friend bool operator==(const MovableOnly& lhs, const MovableOnly& rhs) {return lhs.val == rhs.val;}
private:
    int val;
};

template<class T> using is_my_movable_only_class = std::is_same<MovableOnly, std::remove_cv_t<std::remove_reference_t<T>>>;

struct CopyCounting {
    int move_count;
    int copy_count;

    CopyCounting(): move_count(0), copy_count(0) {}
    CopyCounting(const CopyCounting& rhs): move_count(rhs.move_count), copy_count(rhs.copy_count + 1) {}
    CopyCounting(CopyCounting&& rhs): move_count(rhs.move_count + 1), copy_count(rhs.copy_count) {}
    CopyCounting& operator=(const CopyCounting& rhs) {
        move_count = rhs.move_count;
        copy_count = rhs.copy_count + 1;
        return *this;
    }
    CopyCounting& operator=(CopyCounting&& rhs) {
        move_count = rhs.move_count + 1;
        copy_count = rhs.copy_count;
        return *this;
    }
};

template<class T> using is_my_copy_counting_class = std::is_same<CopyCounting, std::remove_cv_t<std::remove_reference_t<T>>>;

namespace test_extract_arg_by_filtered_index {
    class MyClass {};

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex) {
        auto a1 = extract_arg_by_filtered_index<std::is_integral, 0>(3, "bla", MyClass(), 4, nullptr, 5);
        auto a2 = extract_arg_by_filtered_index<std::is_integral, 1>(3, "bla", MyClass(), 4, nullptr, 5);
        auto a3 = extract_arg_by_filtered_index<std::is_integral, 2>(3, "bla", MyClass(), 4, nullptr, 5);
        EXPECT_EQ(3, a1);
        EXPECT_EQ(4, a2);
        EXPECT_EQ(5, a3);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_singleInput) {
        MyClass obj;
        auto a1 = extract_arg_by_filtered_index<std::is_integral, 0>(3);
        EXPECT_EQ(3, a1);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_movableOnly) {
        MovableOnly a1 = extract_arg_by_filtered_index<is_my_movable_only_class, 0>(3, MovableOnly(3), "test", MovableOnly(1));
        MovableOnly a2 = extract_arg_by_filtered_index<is_my_movable_only_class, 1>(3, MovableOnly(3), "test", MovableOnly(1));
        EXPECT_EQ(MovableOnly(3), a1);
        EXPECT_EQ(MovableOnly(1), a2);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_onlyCopiesIfNecessary) {
        CopyCounting source;
        CopyCounting source2;
        CopyCounting a1 = extract_arg_by_filtered_index<is_my_copy_counting_class, 0>(3, CopyCounting(), "test", source, std::move(source2));
        CopyCounting a2 = extract_arg_by_filtered_index<is_my_copy_counting_class, 1>(3, CopyCounting(), "test", source, std::move(source2));
        CopyCounting a3 = extract_arg_by_filtered_index<is_my_copy_counting_class, 2>(3, CopyCounting(), "test", source, std::move(source2));
        EXPECT_EQ(1, a1.move_count);
        EXPECT_EQ(0, a1.copy_count);
        EXPECT_EQ(0, a2.move_count);
        EXPECT_EQ(1, a3.move_count);
        EXPECT_EQ(0, a3.copy_count);
        EXPECT_EQ(1, a2.copy_count);
    }

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_onlyMovesIfNecessary) {
        CopyCounting source;
        CopyCounting source2;
        CopyCounting&& a1 = extract_arg_by_filtered_index<is_my_copy_counting_class , 0>(3, std::move(source), "test", std::move(source2));
        CopyCounting a2 = extract_arg_by_filtered_index<is_my_copy_counting_class , 1>(3, std::move(source), "test", std::move(source2));
        EXPECT_EQ(0, a1.move_count);
        EXPECT_EQ(0, a1.copy_count);
        EXPECT_EQ(1, a2.move_count);
        EXPECT_EQ(0, a2.copy_count);
    }

    template<class T> using is_true = std::true_type;

    TEST(MetaprogrammingTest, ExtractArgByFilteredIndex_keepsLValueReferencesIntact) {
        MyClass obj;
        MyClass& a1 = extract_arg_by_filtered_index<is_true, 1>(3, obj, "test", obj);
        EXPECT_EQ(&obj, &a1);
    }
}

namespace test_filter_map {
    class MyClass {};

    TEST(MetaprogrammingTest, FilterMap) {
        auto result = filter_map<double, std::is_integral>([] (auto a) {return (double)a;}, 3, "bla", MyClass(), 4, nullptr, 5);
        static_assert(std::is_same<std::array<double, 3>, decltype(result)>::value, "");
        constexpr std::array<double, 3> expected{{3.0, 4.0, 5.0}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_emptyInput) {
        auto result = filter_map<double, std::is_integral>([] (auto a) {return (double)a;});
        static_assert(std::is_same<std::array<double, 0>, decltype(result)>::value, "");
        constexpr std::array<double, 0> expected{{}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_emptyOutput) {
        auto result = filter_map<double, std::is_integral>([] (auto a) {return (double)a;}, "bla", MyClass(), nullptr);
        static_assert(std::is_same<std::array<double, 0>, decltype(result)>::value, "");
        constexpr std::array<double, 0> expected{{}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_movableOnly_byRValue) {
        auto result = filter_map<MovableOnly, is_my_movable_only_class>([] (MovableOnly&& v) {return std::move(v);}, MovableOnly(5), "bla", nullptr, 3, MovableOnly(2));
        static_assert(std::is_same<std::array<MovableOnly, 2>, decltype(result)>::value, "");
        constexpr std::array<MovableOnly, 2> expected {{MovableOnly(5), MovableOnly(2)}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_movableOnly_byValue) {
        auto result = filter_map<MovableOnly, is_my_movable_only_class>([] (MovableOnly v) {return v;}, MovableOnly(5), "bla", nullptr, 3, MovableOnly(2));
        static_assert(std::is_same<std::array<MovableOnly, 2>, decltype(result)>::value, "");
        constexpr std::array<MovableOnly, 2> expected {{MovableOnly(5), MovableOnly(2)}};
        EXPECT_EQ(expected, result);
    }

    TEST(MetaprogrammingTest, FilterMap_onlyCopiesIfNecessary) {
        CopyCounting source;
        CopyCounting source2;
        auto result = filter_map<CopyCounting, is_my_copy_counting_class>([] (CopyCounting v) {return v;}, CopyCounting(), "bla", nullptr, 3, source, std::move(source2));
        static_assert(std::is_same<std::array<CopyCounting, 3>, decltype(result)>::value, "");
        EXPECT_EQ(0, result[0].copy_count);
        EXPECT_EQ(2, result[0].move_count);
        EXPECT_EQ(1, result[1].copy_count);
        EXPECT_EQ(1, result[1].move_count);
        EXPECT_EQ(0, result[2].copy_count);
        EXPECT_EQ(2, result[2].move_count);
    }

    TEST(MetaprogrammingTest, FilterMap_onlyMovesIfNecessary_1) {
        CopyCounting source;
        auto result = filter_map<CopyCounting, is_my_copy_counting_class>([] (CopyCounting&& v) {return std::move(v);}, CopyCounting(), "bla", nullptr, 3, std::move(source));
        static_assert(std::is_same<std::array<CopyCounting, 2>, decltype(result)>::value, "");
        EXPECT_EQ(0, result[0].copy_count);
        EXPECT_EQ(1, result[0].move_count);
        EXPECT_EQ(0, result[1].copy_count);
        EXPECT_EQ(1, result[1].move_count);
    }

    TEST(MetaprogrammingTest, FilterMap_onlyMovesIfNecessary_2) {
        CopyCounting source1;
        CopyCounting source2;
        auto result = filter_map<const CopyCounting*, is_my_copy_counting_class>([] (const CopyCounting& v) {return &v;}, "bla", nullptr, 3, source1, std::move(source2));
        static_assert(std::is_same<std::array<const CopyCounting*, 2>, decltype(result)>::value, "");
        EXPECT_EQ(0, result[0]->copy_count);
        EXPECT_EQ(0, result[0]->move_count);
        EXPECT_EQ(0, result[1]->copy_count);
        EXPECT_EQ(0, result[1]->move_count);
    }
}

namespace test_is_equality_comparable {
    class NotEqualityComparable {};
    class EqualityComparable {};

    inline bool operator==(const EqualityComparable &, const EqualityComparable &) { return false; }

    static_assert(!is_equality_comparable<NotEqualityComparable>::value, "");
    static_assert(is_equality_comparable<EqualityComparable>::value, "");
    static_assert(is_equality_comparable<int>::value, "");
}

namespace test_is_hashable {
    class NotHashable {};
    class Hashable {};
}
}
namespace std {
    template<> struct hash<test_is_hashable::Hashable> final {
        size_t operator()(const test_is_hashable::Hashable &) { return 0; }
    };
}
namespace {
namespace test_is_hashable {
    static_assert(is_hashable<int>::value, "");
    static_assert(is_hashable<Hashable>::value, "");
    static_assert(!is_hashable<NotHashable>::value, "");
}

namespace test_is_function_type {
    class MyClass {};
    class Functor {
        void operator()() {}
    };
    auto lambda = [] () {};

    static_assert(is_function_type<void()>::value, "");
    static_assert(is_function_type<int()>::value, "");
    static_assert(is_function_type<MyClass()>::value, "");
    static_assert(is_function_type<void(MyClass)>::value, "");
    static_assert(is_function_type<void(int)>::value, "");
    static_assert(is_function_type<void(void*)>::value, "");
    static_assert(is_function_type<int()>::value, "");
    static_assert(is_function_type<int(MyClass)>::value, "");
    static_assert(is_function_type<int(const MyClass&)>::value, "");
    static_assert(is_function_type<int(MyClass&&)>::value, "");
    static_assert(is_function_type<MyClass&&()>::value, "");
    static_assert(is_function_type<MyClass&&(MyClass&&)>::value, "");
    static_assert(is_function_type<const MyClass&(int, float, MyClass)>::value, "");

    static_assert(!is_function_type<void>::value, "");
    static_assert(!is_function_type<int>::value, "");
    static_assert(!is_function_type<MyClass>::value, "");
    static_assert(!is_function_type<void*>::value, "");
    static_assert(!is_function_type<const MyClass&>::value, "");
    static_assert(!is_function_type<MyClass&&>::value, "");

    static_assert(!is_function_type<void (*)()>::value, "function pointers aren't plain functions");
    static_assert(!is_function_type<Functor>::value, "Functors aren't plain functions");
    static_assert(!is_function_type<decltype(lambda)>::value, "Lambdas aren't plain functions");
}

namespace test_is_instantiation_of {
    class MyClass {};
    template<class T> class Single {};
    template<class T1, class T2> class Double {};
    template<class... T> class Multiple {};

    static_assert(is_instantiation_of<Single, Single<void>>::value, "");
    static_assert(is_instantiation_of<Single, Single<MyClass>>::value, "");
    static_assert(is_instantiation_of<Single, Single<int>>::value, "");
    static_assert(is_instantiation_of<Single, Single<void*>>::value, "");
    static_assert(is_instantiation_of<Single, Single<int*>>::value, "");
    static_assert(is_instantiation_of<Single, Single<const MyClass&>>::value, "");
    static_assert(is_instantiation_of<Single, Single<MyClass&&>>::value, "");
    static_assert(is_instantiation_of<Double, Double<int, void>>::value, "");
    static_assert(is_instantiation_of<Double, Double<const int&, MyClass*>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<int>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<MyClass&, int>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<MyClass&, int, MyClass>>::value, "");
    static_assert(is_instantiation_of<Multiple, Multiple<MyClass&, int, MyClass, void*>>::value, "");

    static_assert(!is_instantiation_of<Single, Double<int, int>>::value, "");
    static_assert(!is_instantiation_of<Single, Double<int, void>>::value, "");
    static_assert(!is_instantiation_of<Single, Multiple<int>>::value, "");
    static_assert(!is_instantiation_of<Double, Single<int>>::value, "");
    static_assert(!is_instantiation_of<Double, Multiple<int, int>>::value, "");
    static_assert(!is_instantiation_of<Double, Multiple<>>::value, "");
    static_assert(!is_instantiation_of<Multiple, Double<int, int>>::value, "");
    static_assert(!is_instantiation_of<Multiple, Single<int>>::value, "");
}

}

/* Additional material
   -
*/

#include "tree_other/segment.hpp"
#include <iostream>
#include <cassert>
#include <functional>
#include <algorithm>
#include <limits>

template <typename T>
struct max {
    using result_type = T;
    result_type operator()(const T& a, const T& b) const {
        return std::max(a, b);
    }
};
using tree_max_int = tree_other::segment<int, max<int>>;
using tree_plus_int = tree_other::segment<int, std::plus<int>>;
using tree_plus_str = tree_other::segment<std::string, std::plus<std::string>>;
using tree_mul_int = tree_other::segment<int, std::multiplies<int>>;
using tree_mod_int = tree_other::segment<int, std::modulus<int>>;
using tree_xor_int = tree_other::segment<int, std::bit_xor<int>>;

int main()
{
    std::cout << "SEGMENT TREE IMPLEMENTATION TESTS" << std::endl;
    // TODO: test out of range cases
    // edge cases
    {
        std::cout << "checking edge cases ...";
        tree_plus_int bit{};
        assert(bit.view() == std::vector<int>(0));
        std::cout << "ok" << std::endl;
    }
    // creation
    {
        std::cout << "checking creation ...";
        tree_plus_int b1{ 8, 3, 6, 2, 1 };
        assert(b1.view() == std::vector<int>({20, 11, 9, 8, 3, 6, 3, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0}));

        tree_plus_int b2{ 1, 3, 5, 7, 9, 11 };
        assert(b2.view() == std::vector<int>({ 36, 9, 27, 1, 8, 7, 20, 0, 0, 3, 5, 0, 0, 9, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
        std::cout << "ok" << std::endl;
    }
    // query
    {
        std::cout << "checking query ...";
        tree_plus_int b1{ 1, 3, 5, 7, 9, 11 };
        assert(query(b1, std::make_pair(2, 4)) == 21);

        tree_max_int b2{ 1, 3, 5, 7, 9, 11 };
        assert(query(b2, std::make_pair(0, 3)) == 7);
        assert(query(b2, std::make_pair(2, 4)) == 9);

        tree_plus_str b3{ {"aa", "bb", "cc", "dd", "ee", "ff"} };
        assert(query(b3, std::make_pair(0, 6)) == "aabbccddeeff");
        assert(query(b3, std::make_pair(2, 4)) == "ccddee");
        assert(query(b3, std::make_pair(0, 7)) == "aabbccddeeff");

        tree_mul_int b4{ 1, 3, 5, 7, 9, 11 };
        assert(query(b4, std::make_pair(0, 5)) == 10395);

        tree_mod_int b5{ 1, 3, 5, 7, 9, 11 };
        assert(query(b5, std::make_pair(0, 5)) == 1);
        assert(query(b5, std::make_pair(1, 3)) == 3);

        tree_xor_int b6{ 1, 3, 5, 7, 9, 11 };
        assert(query(b6, std::make_pair(0, 5)) == 2);
        assert(query(b6, std::make_pair(1, 3)) == 1);
        std::cout << "ok" << std::endl;
    }
    // update
    {
        std::cout << "checking update ...";
        tree_plus_int b1{ 1, 3, 5, 7, 9, 11 };
        update(b1, std::make_pair(1, 1), 10);
        assert(b1.view() == std::vector<int>({ 43, 16, 27, 1, 15, 7, 20, 0, 0, 10, 5, 0, 0, 9, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));

        tree_max_int b2{ 1, 3, 5, 7, 9, 11 };
        update(b2, std::make_pair(1, 1), 10);
        assert(query(b2, std::make_pair(0, 4)) == 10);
        assert(query(b2, std::make_pair(0, 5)) == 11);
        assert(query(b2, std::make_pair(2, 4)) == 9);

        tree_plus_int b3{ 1, 3, 5, 7, 9, 11 };
        update(b3, std::make_pair(1, 3), 10);
        assert(b3.view() == std::vector<int>({ 51, 21, 30, 1, 20, 10, 20, 0, 0, 10, 10, 0, 0, 9, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));

        tree_plus_str b4{ {"aa", "bb", "cc", "dd", "ee", "ff"} };
        update(b4, std::make_pair(1, 3), std::string("k1"));
        assert(query(b4, std::make_pair(0, 6)) == "aak1k1k1eeff");
        assert(query(b4, std::make_pair(1, 2)) == "k1k1");
        std::cout << "ok" << std::endl;
    }
    // lval
    {
        auto v = { "aa", "bb", "cc", "dd", "ee", "ff" };
        tree_plus_str b4{ { "aa", "bb", "cc", "dd", "ee", "ff" } };
        auto i1 = std::make_pair(1, 1);
        auto s1 = std::string("k1");
        update(b4, i1, s1);
        assert(query(b4, i1) == "k1");
        std::cout << "ok" << std::endl;
    }
}

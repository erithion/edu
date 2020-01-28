/* Additional material
   - https://cp-algorithms.com/data_structures/segment_tree.html
   - https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/
*/

#include <vector>
#include <initializer_list>
#include <type_traits>

namespace tree_other {

    template <typename T, typename Functional, bool Enable = std::is_arithmetic_v<T>>
    struct segment; // Binary indexed tree

    template <typename T, typename F>
    struct segment<T, F, false> {
        static_assert(std::is_arithmetic_v<T>, "This Segment tree does not implement logic for any type except arithmetic");
    };

    template <typename T, typename F>
    struct segment<T, F, true> {
        const std::vector<T> vec_;

        segment(std::initializer_list<T>&& ls) 
            : vec_(tree_size(ls.size())) {
        }

        constexpr static size_t tree_size(size_t n) {
            // Height of a binary tree required to fit all n elements: H = ceil [ log2 n ];
            // Hence the number of nodes to fit all n elements in the leafs: 
            //    S = 1 + 2 + 4 + ... + 2 ^ H = { using geometric progression: (a^0-a^(n+1))/(1-a) } = 2^(H+1) - 1 
            // Since 2 ^ (log2 n) = n except for 'ceil' function, we can get rid of calculating log2: 
            //    S = 2 ^ (H + 1) - 1 = 2 * 2 ^ H - 1 = 2 * 2 ^ (ceil [ log2 n ]) - 1 < 2 * 2 * 2 ^ log2 n = 4 * N => S < 4 * N
            return 4*n;
        }
    };

    template <typename T, typename F>
    void query(const segment<T, F, true>& s, size_t l, size_t r);

    template <typename T, typename F>
    void update(segment<T, F, true>& s, size_t l, size_t r, T val);
}
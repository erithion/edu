/* Additional material
   - https://cp-algorithms.com/data_structures/segment_tree.html
   - https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/
*/

#include <vector>
#include <initializer_list>
#include <type_traits>

namespace tree_other {

    template <typename T, bool Enable = std::is_arithmetic_v<T>>
    struct segment; // Binary indexed tree

    template <typename T>
    struct segment<T, false> {
        static_assert(std::is_arithmetic_v<T>, "This Segment tree does not implement logic for any type except arithmetic");
    };

    template <typename T>
    struct segment<T, true> {

    };
}
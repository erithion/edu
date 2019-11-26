/* Additional material
- https://en.wikipedia.org/wiki/Interval_tree#Augmented_tree
*/

#include "bst.hpp"
#include <algorithm>

namespace tree_search {
    namespace interval {
        namespace aux {
        }

        template <typename T>
        struct tree_type { // Hides aux namespace so that ADL would have found the insert-funcs below to properly recolor the root
            aux::tree_type<T> root_;
            aux::tree_type<T>& operator->() {
                return this->root_;
            }
        };
    }
}
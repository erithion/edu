/* Additional material
- https://www.geeksforgeeks.org/interval-tree/
*/
#pragma once

#include "capability_rotate.hpp"
#include "tree_redblack.hpp"

#include <limits>

namespace tree_search {

    namespace interval_aux {

        template <typename T>
        struct interval_augment 
            : public aux::red_black_augment {
            T    max_ = std::numeric_limits<T>::min();
        };
    }
}
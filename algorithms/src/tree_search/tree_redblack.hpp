#pragma once

#include "type_capability.hpp"
#include "capability_rotate.hpp"

#include <memory>

namespace tree_search {

    namespace aux {

        enum rb_color : bool {
            rb_black = false,
            rb_red = true
        };

        template <typename, typename>
        struct redblack_augment {
            rb_color color_ = rb_red;
        };
    }

    // TODO: implement invariants check function like balanced
}
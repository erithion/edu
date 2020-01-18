#pragma once

#include <memory>
#include <functional>
#include <type_traits>

namespace tree_search {
 
    struct capability_insert {};
    namespace aux {

        template <typename T, typename Tree>
        void insert_(std::unique_ptr<Tree>& tree, T&& v, capability_insert) { // universal reference
            if (!tree) tree = std::make_unique<typename Tree::rr_node_type>(std::forward<T>(v));
            else if (v < tree->value_) insert_(tree->left_, std::forward<T>(v), capability_insert{});
            else if (v > tree->value_) insert_(tree->right_, std::forward<T>(v), capability_insert{});
        }
    }

    template <typename T, typename Tree, std::enable_if_t<std::is_base_of_v<capability_insert, Tree>, int> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        aux::insert_(tree.root_, std::forward<T>(v), capability_insert{});
    }

    template <typename T, typename Tree, std::enable_if_t<std::is_base_of_v<capability_insert, Tree>, int> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        for (auto&& v : ls) aux::insert_(tree.root_, std::move(v), capability_insert{});
    }
}
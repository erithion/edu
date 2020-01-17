#pragma once

#include <memory>
#include <functional>
#include <type_traits>

namespace tree_search {
 
    struct insert_cap {};
    namespace aux {

        template <typename T, typename Tree>
        void insert_(std::unique_ptr<Tree>& tree, T&& v, insert_cap) { // universal reference
            if (!tree) tree = std::make_unique<typename Tree::rr_node_type>(std::forward<T>(v));
            else if (v < tree->value_) insert_(tree->left_, std::forward<T>(v), insert_cap{});
            else if (v > tree->value_) insert_(tree->right_, std::forward<T>(v), insert_cap{});
        }
    }

    template <typename T, typename Tree, std::enable_if_t<std::is_base_of_v<insert_cap, Tree>, int> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        aux::insert_(tree.root_, std::forward<T>(v), insert_cap{});
    }

    template <typename T, typename Tree, std::enable_if_t<std::is_base_of_v<insert_cap, Tree>, int> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        for (auto&& v : ls) aux::insert_(tree.root_, std::move(v), insert_cap{});
    }
}
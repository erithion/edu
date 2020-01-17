/* Additional material
- https://www.geeksforgeeks.org/interval-tree/
*/
#pragma once

#include "tree_interval.hpp"

#include <utility>
#include <memory>
#include <algorithm>

namespace tree_search {

    struct interval_insert_cap {};

    namespace interval_aux {

        template <typename Tree>
        void update_max(std::unique_ptr<Tree>& tree) {
            if (!tree) return;
            tree->augment_.max_ = tree->value_.second;
            if (tree->left_)
                tree->augment_.max_ = std::max(tree->left_->augment_.max_, tree->augment_.max_);
            if (tree->right_)
                tree->augment_.max_ = std::max(tree->right_->augment_.max_, tree->augment_.max_);
        }
        //TODO: functions differ only reference type of pair. resolve properly
        template <typename T, typename Tree>
        void insert(std::unique_ptr<Tree>& tree, const std::pair<T, T>& v, interval_insert_cap) { // universal reference
            if (!tree) {
                tree = std::make_unique<typename Tree::rr_node_type>(std::move(v));
                tree->augment_.max_ = v.second;
            }
            else if (v < tree->value_) insert(tree->left_, std::move(v), interval_insert_cap{});
            else if (v > tree->value_) insert(tree->right_, std::move(v), interval_insert_cap{});
            aux::fix_red_black_invariant(tree);
            update_max(tree->left_);
            update_max(tree->right_);
            update_max(tree);
        }

        template <typename T, typename Tree>
        void insert(std::unique_ptr<Tree>& tree, const std::pair<T, T>&& v, interval_insert_cap) { // universal reference
            if (!tree) {
                tree = std::make_unique<typename Tree::rr_node_type>(std::move(v));
                tree->augment_.max_ = v.second;
            }
            else if (v < tree->value_) insert(tree->left_, std::move(v), interval_insert_cap{});
            else if (v > tree->value_) insert(tree->right_, std::move(v), interval_insert_cap{});
            aux::fix_red_black_invariant(tree);
            update_max(tree->left_);
            update_max(tree->right_);
            update_max(tree);
        }
    }

    template <typename T, typename Tree, std::enable_if_t<std::is_base_of_v<interval_insert_cap, Tree>, int> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        interval_aux::insert(tree.root_, std::forward<T>(v), interval_insert_cap{});
        tree.root_->augment_.is_red_ = false;
    }

    template <typename T, typename Tree, std::enable_if_t<std::is_base_of_v<interval_insert_cap, Tree>, int> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        // be mindful that initializers_list iterator produces pointers to const
        for (auto&& v : ls) insert(tree, std::move(v)); // call must go through insert with setting root to black
    }
}
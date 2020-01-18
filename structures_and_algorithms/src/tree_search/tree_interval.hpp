/* Additional material
- https://www.geeksforgeeks.org/interval-tree/
*/
#pragma once

#include "capability_rotate.hpp"
#include "tree_redblack.hpp"

#include <limits>
#include <utility>
#include <memory>
#include <algorithm>

namespace tree_search {

    struct capability_insert_interval {};

    namespace aux {

        template <typename T>
        struct interval_augment 
            : public aux::redblack_augment {
            using value_type = T;
            T    max_ = std::numeric_limits<T>::min();
        };

        // Enables int type if interval tree requirements are satisfied
        template <typename T, typename Tree>
        using enable_interval_insert_t =
            std::enable_if_t< std::is_base_of_v<capability_insert_interval, Tree> // only for capability_insert_interval
                           && std::is_same_v<typename Tree::augment_type, 
                                             aux::interval_augment<typename Tree::augment_type::value_type>> // only for interval_augment
                           && std::is_same_v<std::pair<typename Tree::augment_type::value_type, typename Tree::augment_type::value_type>,
                                             std::remove_reference_t<std::remove_cv_t<T>>> // only for std::pair
            , int>;

        // Compliance within aux namespace is not checked in general. 
        // Just capability_insert_interval is used as a guard to prevent accidental overloading resolution calls
        template <typename Tree>
        void update_max(std::unique_ptr<Tree>& tree) {
            if (!tree) return;
            tree->augment_.max_ = tree->value_.second;
            if (tree->left_)
                tree->augment_.max_ = std::max(tree->left_->augment_.max_, tree->augment_.max_);
            if (tree->right_)
                tree->augment_.max_ = std::max(tree->right_->augment_.max_, tree->augment_.max_);
        }

        template <typename PairT, typename Tree>
        void insert(std::unique_ptr<Tree>& tree, PairT&& v, capability_insert_interval) {
            if (!tree) {
                tree = std::make_unique<typename Tree::rr_node_type>(std::forward<PairT>(v));
                tree->augment_.max_ = v.second;
            }
            else if (v < tree->value_) insert(tree->left_, std::forward<PairT>(v), capability_insert_interval{});
            else if (v > tree->value_) insert(tree->right_, std::forward<PairT>(v), capability_insert_interval{});
            aux::fix_invariant(tree, capability_insert_redblack{});
            update_max(tree->left_);
            update_max(tree->right_);
            update_max(tree);
        }
    }

    template <typename T, typename Tree, aux::enable_interval_insert_t<T, Tree> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        aux::insert(tree.root_, std::forward<T>(v), capability_insert_interval{});
        tree.root_->augment_.is_red_ = false;
    }

    template <typename T, typename Tree, aux::enable_interval_insert_t<T, Tree> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        // be mindful that initializers_list iterator produces pointers to const, so move still leads to copies
        for (auto&& v : ls) insert(tree, std::move(v)); // call must go through insert with setting root to black
    }
}
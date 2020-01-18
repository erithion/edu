/* Additional material
- https://brilliant.org/wiki/avl-tree/?subtopic=types-and-data-structures&chapter=binary-search-trees
- MIT Algorithms course https://www.youtube.com/watch?v=FNeL18KsWPc
*/
#pragma once

#include "capability_rotate.hpp"

#include <memory>
#include <algorithm>

namespace tree_search {

    struct capability_insert_avl {};

    namespace aux {

        struct avl_augment {
            int                height_ = -1;
        };

        template <typename Tree>
        int def_height(std::unique_ptr<Tree>& t) {
            return t ? t->augment_.height_ : -1;
        }

        template <typename Tree>
        void update_height(std::unique_ptr<Tree>& t) {
            if (t) t->augment_.height_ = 1 + std::max(def_height(t->left_), def_height(t->right_));
        }

        template <typename Tree>
        void fix_invariant(std::unique_ptr<Tree>& t, capability_insert_avl) { // AVL invariant: abs(right - left) must be at most 1
            auto dif = def_height(t->right_) - def_height(t->left_);
            if (dif > 1) { // right subtree
                if (def_height(t->right_->left_) > def_height(t->right_->right_)) {
                    // case    \  
                    //         /
                    //
                    rotate(t->right_, rotate_right_tag{});
                    // update only two nodes
                    update_height(t->right_->right_);
                    update_height(t->right_);
                }
                // case        \
                    //              \ 
                    //
                rotate(t, rotate_left_tag{});
                // update only one node; current node will be processed further
                update_height(t->left_);
            }
            else if (dif < -1) {
                if (def_height(t->left_->right_) > def_height(t->left_->left_)) {
                    // case    /  
                    //         \
                        //
                    rotate(t->left_, rotate_left_tag{});
                    // update only two nodes
                    update_height(t->left_->left_);
                    update_height(t->left_);
                }
                // case        /
                //            / 
                //
                rotate(t, rotate_right_tag{});
                // update only one node; current node will be processed further
                update_height(t->right_);
            }
            update_height(t);
        }

        // TODO: make common boilerplate for red-black and avl, call fix_invariant by type 
        template <typename T, typename Tree>
        void insert(std::unique_ptr<Tree>& tree, T&& v, capability_insert_avl) { // universal reference
            if (!tree) tree = std::make_unique<typename Tree::rr_node_type>(std::forward<T>(v));
            else if (v < tree->value_) insert(tree->left_, std::forward<T>(v), capability_insert_avl{});
            else if (v > tree->value_) insert(tree->right_, std::forward<T>(v), capability_insert_avl{});
            fix_invariant(tree, capability_insert_avl{});
        }

        // Enables int type if avl tree requirements are satisfied
        template <typename T, typename Tree>
        using enable_avl_insert_t =
            std::enable_if_t< std::is_base_of_v<capability_insert_avl, Tree> // only for capability_insert_avl
                           && std::is_same_v<typename Tree::augment_type, aux::avl_augment> // only for avl_augment
                           && !std::is_base_of_v<capability_rotate, Tree> // rotation available on a user side may invalidate invariants
            , int>;
    }

    template <typename T, typename Tree, aux::enable_avl_insert_t<T, Tree> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        aux::insert(tree.root_, std::forward<T>(v), capability_insert_avl{});
    }

    template <typename T, typename Tree, aux::enable_avl_insert_t<T, Tree> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        for (auto&& v : ls) insert(tree, std::move(v));
    }

}
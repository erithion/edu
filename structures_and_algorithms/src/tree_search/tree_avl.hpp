/* Additional material
- https://brilliant.org/wiki/avl-tree/?subtopic=types-and-data-structures&chapter=binary-search-trees
- MIT Algorithms course https://www.youtube.com/watch?v=FNeL18KsWPc
*/
#pragma once

#include "capability_rotate.hpp"

#include <memory>
#include <algorithm>

namespace tree_search {

    struct avl_insert_cap {};

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
        void fix_avl_invariant(std::unique_ptr<Tree>& t) { // AVL invariant: abs(right - left) must be at most 1
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
        void avl_insert(std::unique_ptr<Tree>& tree, T&& v) { // universal reference
            if (!tree) tree = std::make_unique<typename Tree::rr_node_type>(std::forward<T>(v));
            else if (v < tree->value_) avl_insert(tree->left_, std::forward<T>(v));
            else if (v > tree->value_) avl_insert(tree->right_, std::forward<T>(v));
            fix_avl_invariant(tree);
        }
    }

    // TODO: add check for avl augmented struct presence
    template <typename T, typename Tree, std::enable_if_t< std::is_base_of_v<avl_insert_cap, Tree>
                                                        && std::is_base_of_v<rotate_cap, Tree>, int> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        aux::avl_insert(tree.root_, std::forward<T>(v));
    }

    // TODO: add check for avl augmented struct presence
    template <typename T, typename Tree, std::enable_if_t< std::is_base_of_v<avl_insert_cap, Tree>
                                                        && std::is_base_of_v<rotate_cap, Tree>, int> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        for (auto&& v : ls) insert(tree, std::move(v));
    }

}
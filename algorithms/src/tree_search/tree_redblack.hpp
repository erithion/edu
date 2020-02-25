/* Additional material
- https://brilliant.org/wiki/red-black-tree/?subtopic=types-and-data-structures&chapter=binary-search-trees  (insert only. deletion is wrong there.)
- MIT Course https://www.youtube.com/watch?v=O3hI9FdxFOM
- Visualisation for testing http://www.cs.armstrong.edu/liang/animation/web/RBTree.html

- https://www.youtube.com/watch?v=CTvfzU_uNKE deletion (Tushar Roy)
*/
#pragma once

#include "capability_rotate.hpp"
#include "capability_insert.hpp"
#include "capability_remove.hpp"
#include "type_traverse.hpp" // TODO: move is_one_of out
#include "iterator_aux.hpp" // TODO: move out node_type_t

#include <memory>
#include <algorithm>
#include <cassert>

namespace tree_search {

    struct capability_insert_redblack {};
    struct capability_remove_redblack {};

    namespace aux {

        struct redblack_augment {
            bool                is_red_ = true;
        };

        // Compliance within aux namespace is not checked in general. 
        // Just capability_insert_redblack is used on and off as a guard to prevent accidental overloading resolution calls
        template <typename Node>
        inline bool is_left_red(const std::unique_ptr<Node>& tree) {
            return tree && tree->left_ && tree->left_->is_red_;
        }

        template <typename Node>
        inline bool is_right_red(const std::unique_ptr<Node>& tree) {
            return tree && tree->right_ && tree->right_->is_red_;
        }

        template <typename Node>
        inline bool any_red_child(const std::unique_ptr<Node>& tree) {
            return is_left_red(tree) || is_right_red(tree);
        }

        template <typename Node>
        inline bool any_red_grandchild(const std::unique_ptr<Node>& tree) {
            if (!tree) return false;
            return is_left_red(tree->left_) || is_right_red(tree->left_)
                || is_left_red(tree->right_) || is_right_red(tree->right_);
        }

        // is called from insert
        template <typename Node>
        void fix_invariant(std::unique_ptr<Node>& tree, const capability_insert_redblack&) {
            if (tree && !tree->is_red_) {
                if (is_left_red(tree) && is_right_red(tree) && any_red_grandchild(tree)) { // Case 1 (left & right) -> recolor
                    tree->is_red_ = true;
                    tree->left_->is_red_ = false;
                    tree->right_->is_red_ = false;
                }
                else if (is_left_red(tree) && any_red_child(tree->left_)) {
                    if (is_right_red(tree->left_)) // case 2 - straightening zig-zag
                        rotate(tree->left_, rotate_left_tag{});
                    rotate(tree, rotate_right_tag{}); // case 3
                    //
                    tree->is_red_ = false;
                    tree->right_->is_red_ = true;
                }
                else if (is_right_red(tree) && any_red_child(tree->right_)) { // symmetric to previous
                    if (is_left_red(tree->right_)) // case 2 - straightening zig-zag
                        rotate(tree->right_, rotate_right_tag{});
                    rotate(tree, rotate_left_tag{}); // case 3
                    //
                    tree->is_red_ = false;
                    tree->left_->is_red_ = true;
                }
            }
        }

        template <typename Node>
        bool fix_invariant_case4(std::unique_ptr<Node>& tree, std::unique_ptr<Node>& parent, const capability_remove_redblack&) {
            Node* other = parent->left_ == tree ? parent->right_.get() : parent->left_.get();
            bool flag = parent->is_red_
                && other && !other->is_red_
                && (!other->left_ /*sentinel node, hence black*/ || !other->left_->is_red_ /*not sentinel and black*/)
                && (!other->right_ || !other->right_->is_red_);
            if (flag)
                std::swap(parent->is_red_, other->is_red_);
            return flag;
        }

        template <typename Node>
        bool fix_invariant_case6(std::unique_ptr<Node>& tree, std::unique_ptr<Node>& parent, const capability_remove_redblack&) {
            Node* other = parent->left_ == tree ? parent->right_.get() : parent->left_.get();
            bool flag = other && !other->is_red_
                && (other->right_ && other->right_->is_red_);
            if (flag) {
                std::swap(parent->is_red_, other->is_red_);
                other->right_->is_red_ = false;
                if (parent->left_ == tree) rotate(parent, rotate_left_tag{});
                else rotate(parent, rotate_right_tag{});
            }
            return flag;
        }

        template <typename Node>
        bool fix_invariant(std::unique_ptr<Node>& tree, std::unique_ptr<Node>& parent, const capability_remove_redblack&) {
        }


        template <typename T, typename Node>
        void remove(std::unique_ptr<Node>& tree, std::unique_ptr<Node>& parent, const T& v, const capability_remove_redblack& tag) {
            if (!tree) return;
            else if (v < tree->value_) remove(tree->left_, tree, v, tag);
            else if (v > tree->value_) remove(tree->right_, tree, v, tag);
            else { // found
                auto p = std::ref(tree);
                auto prev = std::ref(parent);
                if (tree->left_ && tree->right_) {
                    for (p = tree->right_; p.get()->left_; prev = p, p = p.get()->left_);
                    tree->value_ = std::move(p.get()->value_);
                }
                else if (tree->left_) {
                    tree->value_ = std::move(tree->left_->value_);
                    p = tree->left_;
                }
                else if (tree->right_) {
                    tree->value_ = std::move(tree->right_->value_);
                    p = tree->right_;
                }
                else {
                    if (!tree->is_red_) {
                        if (fix_invariant_case4(tree, parent, tag)) {}// exit with node deletion;
                        else if (fix_invariant_case6(tree, parent, tag)) {} // exit with node deletion;
                    }
//                    assert(("black node is about to be deleted. implement it", tree->augment_.is_red_));
                    tree = nullptr;
                    return;
                }
                remove(p.get(), prev.get(), p.get()->value_, tag);
            }
        }

        // Enables int type if red-black tree requirements are satisfied
        template <typename T, typename Tree>
        using enable_redblack_insert_t =
            std::enable_if_t< std::is_base_of_v<capability_insert_redblack, Tree> // only for capability_insert_redblack
                            && std::is_same_v<typename Tree::augment_type, aux::redblack_augment> // only for redblack_augment
                            && !std::is_base_of_v<capability_rotate, Tree> // rotation available on a user side may invalidate invariants
            , int>;
        // Enables int type if red-black tree requirements are satisfied
        template <typename T, typename Tree>
        using enable_redblack_remove_t =
            std::enable_if_t< std::is_base_of_v<capability_remove_redblack, Tree> // only for capability_remove_redblack
                            && std::is_same_v<typename Tree::augment_type, aux::redblack_augment> // only for redblack_augment
                            && !std::is_base_of_v<capability_rotate, Tree> // rotation available on a user side may invalidate invariants
            , int>;
    }

    // TODO: verify against an existing elements insertion
    // TODO: implement remove
    template <typename T, typename Tree, aux::enable_redblack_insert_t<T, Tree> = 0>
    void insert(Tree& tree, T&& v) { // universal reference
        aux::insert(aux::access(tree), std::forward<T>(v), capability_insert_redblack{});
        aux::access(tree)->is_red_ = false;
    }

    template <typename T, typename Tree, aux::enable_redblack_insert_t<T, Tree> = 0>
    void insert(Tree& tree, std::initializer_list<T> ls) {
        // be mindful that initializers_list iterator produces refs to const, so move still leads to copies
        for (auto&& v : ls) insert(tree, std::move(v)); // call must go through insert with setting root to black
    }

    template <typename T, typename Tree, aux::enable_redblack_remove_t<T, Tree> = 0>
    void remove(Tree& tree, const T& v) {
        aux::ptr_type_t<Tree> null{};
        aux::remove(aux::access(tree), null, v, capability_remove_redblack{});
        aux::access(tree)->is_red_ = false;
    }

    template <typename T, typename Tree, aux::enable_redblack_remove_t<T, Tree> = 0>
    void remove(Tree& tree, std::initializer_list<T> ls) {
        for (auto&& v : ls) remove(tree, v);
    }
}
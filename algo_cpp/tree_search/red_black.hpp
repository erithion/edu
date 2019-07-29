/* Additional material
- https://brilliant.org/wiki/red-black-tree/?subtopic=types-and-data-structures&chapter=binary-search-trees
- MIT Course https://www.youtube.com/watch?v=O3hI9FdxFOM
- Visualisation for testing http://www.cs.armstrong.edu/liang/animation/web/RBTree.html
*/

#include "bst.hpp"
#include <algorithm>

namespace tree_search {
    namespace red_black {
        namespace aux {
            using tree_search::aux::find;
            using tree_search::aux::remove;
            using tree_search::aux::size;
            using tree_search::aux::traverse;

            using tree_search::aux::inorder_tag;
            using tree_search::aux::preorder_tag;
            using tree_search::aux::postorder_tag;

            struct ext {
                bool is_red_ = true;
            };
            // For the user of BST
            template <typename T>
            using tree_type = std::unique_ptr<tree_search::aux::base::node<T, ext>>;

            template <typename T>
            using tree_typec = std::unique_ptr<tree_search::aux::base::node<std::remove_cv_t<T>, ext>>;

            template <typename T>
            bool is_left_red(tree_type<T>& tree) {
                return tree && tree->left_ && tree->left_->is_red_;
            }

            template <typename T>
            bool is_right_red(tree_type<T>& tree) {
                return tree && tree->right_ && tree->right_->is_red_;
            }

            template <typename T>
            bool any_red_child(tree_type<T>& tree) {
                return is_left_red(tree) || is_right_red(tree);
            }

            template <typename T>
            bool any_red_grandchild(tree_type<T>& tree) {
                if (!tree) return false;
                return is_left_red(tree->left_) || is_right_red(tree->left_)
                    || is_left_red(tree->right_) || is_right_red(tree->right_);
            }

            /* Props of the red-black tree
            1. Each node is either red or black.
            2. The root is black. 
            3. All leaves (NIL) are black.
            4. If a node is red, then its parent is black.
            5. Every path from a given node to any of its descendant NIL nodes contains the same number of black nodes (black-depth)
            6. The height of the red-black tree is at most 2 * log2(n + 1)
            */
            template <typename T>
            void fix_red_black_invariant(tree_type<T>& tree) {
                if (tree && !tree->is_red_) {
                    if (is_left_red(tree) && is_right_red(tree) && any_red_grandchild(tree)) { // Case 1 (left & right) -> recolor
                        tree->is_red_ = true;
                        tree->left_->is_red_ = false;
                        tree->right_->is_red_ = false;
                    }
                    else if (is_left_red(tree) && any_red_child(tree->left_)) {
                        if (is_right_red(tree->left_)) // case 2 - straightening zig-zag
                            rotate(tree_search::aux::left_tag{}, tree->left_);
                        rotate(tree_search::aux::right_tag{}, tree); // case 3
                        //
                        tree->is_red_ = false;
                        tree->right_->is_red_ = true;
                    }
                    else if (is_right_red(tree) && any_red_child(tree->right_)) { // symmetric to previous
                        if (is_left_red(tree->right_)) // case 2 - straightening zig-zag
                            rotate(tree_search::aux::right_tag{}, tree->right_);
                        rotate(tree_search::aux::left_tag{}, tree); // case 3
                        //
                        tree->is_red_ = false;
                        tree->left_->is_red_ = true;
                    }
                }
            }

            template <typename T>
            void insert(tree_typec<T>& t, T&& v) {
                std::function<void(tree_typec<T>&)> fn = [](tree_typec<T>& t) {
                    fix_red_black_invariant(t);
                };
                tree_search::aux::insert(t, std::forward<T>(v), fn); // default implementation
            }
        }

        template <typename T>
        struct tree_type { // Hides aux namespace so that ADL would have found the insert-funcs below to properly recolor the root
            aux::tree_type<T> root_;
            aux::tree_type<T>& operator->() {
                return this->root_;
            }
        };

        template <typename T>
        void insert(tree_type<std::remove_cv_t<T>>& t, T&& v) {
            aux::insert(t.root_, std::forward<T>(v));
            // recolor root to satisfy prop #2 
            t.root_->is_red_ = false;
        }

        template <typename T>
        void insert(tree_type<std::remove_cv_t<T>>& tree, std::initializer_list<T> ls) {
            for (auto&& v : ls) insert(tree, std::move(v));
        }

        template <typename T>
        auto size(tree_type<T>& tree) {
            return size(tree.root_);
        }

        //        using tree_search::aux::find;
        //        using tree_search::aux::remove;

        using tree_search::aux::inorder_tag;
        using tree_search::aux::preorder_tag;
        using tree_search::aux::postorder_tag;

        template <typename T, typename TraverseType, typename CallbackType>
        void traverse(TraverseType&& trav, const tree_type<T>& tree, CallbackType&& fn) {
            tree_search::aux::traverse(std::forward<TraverseType>(trav), tree.root_, std::forward<CallbackType>(fn));
        }
    }
}
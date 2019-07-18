/* Additional material
- https://brilliant.org/wiki/avl-tree/?subtopic=types-and-data-structures&chapter=binary-search-trees
- MIT Algorithms course https://www.youtube.com/watch?v=FNeL18KsWPc
*/

#include "tree_binary.hpp"
#include <algorithm>

namespace bst {
    namespace avl {
        namespace aux {
            using bst::aux::find;
            using bst::aux::remove;
            using bst::aux::size;
            using bst::aux::traverse;

            using bst::aux::inorder_tag;
            using bst::aux::preorder_tag;
            using bst::aux::postorder_tag;

            struct ext {
                int height_ = -1;
            };
            // For the user of BST
            template <typename T>
            using tree_type = std::unique_ptr<bst::aux::base::node<T, ext>>;

            template <typename T>
            using tree_typec = std::unique_ptr<bst::aux::base::node<std::remove_cv_t<T>, ext>>;

            template <typename T>
            int def_height(tree_type<T>& t) {
                return t ? t->height_ : -1;
            }

            template <typename T>
            void update_height(tree_type<T>& t) {
                if (t) t->height_ = 1 + std::max(def_height(t->left_), def_height(t->right_));
            }

            template <typename T>
            void fix_avl_invariant(tree_type<T>& t) { // AVL invariant: abs(right - left) must be at most 1
                auto dif = def_height(t->right_) - def_height(t->left_);
                if (dif > 1) { // right subtree
                    if (def_height(t->right_->left_) > def_height(t->right_->right_)) {
                        // case    \  
                        //         /
                        //
                        rotate(bst::aux::right_tag{}, t->right_);
                        // update only two nodes
                        update_height(t->right_->right_);
                        update_height(t->right_);
                    }
                    // case        \
                    //              \ 
                    //
                    rotate(bst::aux::left_tag{}, t);
                    // update only one node; current node will be processed further
                    update_height(t->left_);
                }
                else if (dif < -1) {
                    if (def_height(t->left_->right_) > def_height(t->left_->left_)) {
                        // case    /  
                        //         \
                        //
                        rotate(bst::aux::left_tag{}, t->left_);
                        // update only two nodes
                        update_height(t->left_->left_);
                        update_height(t->left_);
                    }
                    // case        /
                    //            / 
                    //
                    rotate(bst::aux::right_tag{}, t);
                    // update only one node; current node will be processed further
                    update_height(t->right_);
                }
                update_height(t);
            }

            template<typename T>
            struct CTTI; // compile time type inspector; just generates a compile time error with a specified type T

            template <typename T>
            void insert(tree_typec<T>& t, T&& v) {
//                CTTI<T> a; 
//                CTTI<decltype(t)> b;
                std::function<void(tree_typec<T>&)> fn = [](tree_typec<T>& t) {
                    fix_avl_invariant(t); 
                };
                bst::aux::insert(t, std::forward<T>(v), fn); // default implementation
            }

            // importing default templatized implementation for initializer_list
            // for inserting one element ADL will resort to the specialized insert above
            using bst::aux::insert;
        }
        using bst::avl::aux::insert;
        using bst::aux::find;
        using bst::aux::remove;
        using bst::aux::size;
        using bst::aux::traverse;

        using bst::aux::inorder_tag;
        using bst::aux::preorder_tag;
        using bst::aux::postorder_tag;
        using bst::avl::aux::tree_type;
    }
}
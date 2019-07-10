/* Additional material
- https://stackoverflow.com/questions/14001676/balancing-a-bst
- Day-Warren-Stout algorithm http://penguin.ewu.edu/~trolfe/DSWpaper/
*/

#include "tree_binary.hpp"

namespace bst {
    namespace binary {
        using bst::aux::insert;
        using bst::aux::find;
        using bst::aux::remove;
        using bst::aux::size;
        using bst::aux::traverse;

        using bst::aux::inorder_tag;
        using bst::aux::preorder_tag;
        using bst::aux::postorder_tag;

        namespace aux {
            using bst::aux::insert;
            using bst::aux::find;
            using bst::aux::remove;
            using bst::aux::size;
            using bst::aux::traverse;
            using bst::aux::rotate;

            using bst::aux::right_tag;
            using bst::aux::left_tag;

            using bst::aux::inorder_tag;
            using bst::aux::preorder_tag;
            using bst::aux::postorder_tag;

            struct ext {};
            // For the user of BST
            template <typename T>
            using tree_type = std::unique_ptr<bst::aux::base::node<T, ext>>;

            // Transforms a tree into a right degenerate tree (or "vine" according to the Warren-Stout algorithm's notation)
            template <typename Tree>
            size_t to_right_list(tree_type<Tree>& tree) {
                size_t size = 0;
                auto   cur = std::ref(tree);
                while (cur.get())
                    if (cur.get()->left_) rotate(right_tag{}, cur.get());
                    else ++size, cur = std::ref(cur.get()->right_);
                return size;
            }

            // Re-balance a tree to a COMPLETE tree by the Warren-Stout algorithm 
            template <typename Tree>
            void balance(tree_type<Tree>& tree) {
                auto size = to_right_list(tree);
                // We want the highest FULL tree size not bigger than the current size: n_full_highest <= size
                // Geometric progression gives n_full = 2^0 + 2^1 + 2^2 + ... + 2^k = 2^(k+1)-1 = 2^height - 1 => n_full + 1 = 2^height
                // Since any FULL tree size ultimately results in powers of 2, 
                // it's enough to find out the lower (power of 2) border of the current nonfull tree size
                // log2 and floor will do the trick
                auto height = static_cast<size_t>(std::floor(std::log2(size + 1)));
                auto n_full = static_cast<size_t>(std::pow(2, height) - 1); // -1 is due to the above formula for n_full 
                // All that in order to generate a COMPLETE tree, i.e. leftovers beyond n_full must be the leftmost elements of the tree
                // So we left-rotate the leftovers first
                auto leftovers_count = size - n_full;
                auto cur = std::ref(tree);
                for (auto j = 0; j != leftovers_count; ++j) {
                    rotate(left_tag{}, cur.get());
                    cur = std::ref(cur.get()->right_);
                }
                // Now just balance
                for (auto i = n_full / 2; i != 0; i /= 2) {
                    auto cur = std::ref(tree);
                    for (auto j = 0; j != i; ++j) {
                        rotate(left_tag{}, cur.get());
                        cur = std::ref(cur.get()->right_);
                    }
                }
            }

            template <typename Tree>
            bool is_balanced(tree_type<Tree>& root);
        }
        using bst::binary::aux::tree_type;
        using bst::binary::aux::balance;
    }
}
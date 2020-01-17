/* Additional material
*/

#include "tree_search/tree_redblack.hpp"
#include "tree_search/capability_traverse.hpp"
#include "tree_search/tree.hpp"
#include <cassert>
#include <iostream>

namespace ts = tree_search;

template <typename T>
using tree = ts::tree<T, ts::aux::red_black_augment, ts::traverse_cap, ts::rb_insert_cap>;

int main()
{
    std::cout << "RED-BLACK TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking red-black tree size ...";
        tree<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        assert(size(bin) == 7);
        std::cout << "ok" << std::endl;
    }
    // checking inorder traversing
    {
        std::cout << "checking inorder traversing ...";
        tree<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        std::vector<int> res;
        traverse(ts::tag_inorder{}, bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<int>({ 4, 5, 7, 8, 11, 15, 16 }));
        std::cout << "ok" << std::endl;
        // rotate(ts::rotate_right_tag{}, bin); // must emit compile time error for rotate_cap has not been added into the tree
    }
    //
    {
        std::cout << "checking red-black tree invariant ...";
        tree<int> bin;
        insert(bin, {11, 5, 15, 4, 7, 16, 8});
        assert(bin.root_->value_ == 11);
        assert(bin.root_->augment_.is_red_ == false);
        assert(bin.root_->left_->value_ == 5);
        assert(bin.root_->left_->augment_.is_red_ == true);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->left_->augment_.is_red_ == false);
        assert(bin.root_->left_->right_->value_ == 7);
        assert(bin.root_->left_->right_->augment_.is_red_ == false);
        assert(bin.root_->left_->right_->right_->value_ == 8);
        assert(bin.root_->left_->right_->right_->augment_.is_red_ == true);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->augment_.is_red_ == false);
        assert(bin.root_->right_->right_->value_ == 16);
        assert(bin.root_->right_->right_->augment_.is_red_ == true);

        insert(bin, 9);
        assert(bin.root_->value_ == 11);
        assert(bin.root_->augment_.is_red_ == false);
        assert(bin.root_->left_->value_ == 5);
        assert(bin.root_->left_->augment_.is_red_ == true);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->left_->augment_.is_red_ == false);
        assert(bin.root_->left_->right_->value_ == 8);
        assert(bin.root_->left_->right_->augment_.is_red_ == false);
        assert(bin.root_->left_->right_->right_->value_ == 9);
        assert(bin.root_->left_->right_->right_->augment_.is_red_ == true);
        assert(bin.root_->left_->right_->left_->value_ == 7);
        assert(bin.root_->left_->right_->left_->augment_.is_red_ == true);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->augment_.is_red_ == false);
        assert(bin.root_->right_->right_->value_ == 16);
        assert(bin.root_->right_->right_->augment_.is_red_ == true);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking red-black tree insertion against MIT example covering all 3 cases ...";
        tree<int> bin;
        insert(bin, { 7, 3, 18, 10, 22, 8, 11, 26 });
        assert(bin.root_->value_ == 7);
        assert(bin.root_->augment_.is_red_ == false);
        assert(bin.root_->left_->value_ == 3);
        assert(bin.root_->left_->augment_.is_red_ == false);
        assert(bin.root_->right_->value_ == 18);
        assert(bin.root_->right_->augment_.is_red_ == true);
        assert(bin.root_->right_->left_->value_ == 10);
        assert(bin.root_->right_->left_->augment_.is_red_ == false);
        assert(bin.root_->right_->left_->left_->value_ == 8);
        assert(bin.root_->right_->left_->left_->augment_.is_red_ == true);
        assert(bin.root_->right_->left_->right_->value_ == 11);
        assert(bin.root_->right_->left_->right_->augment_.is_red_ == true);
        assert(bin.root_->right_->right_->value_ == 22);
        assert(bin.root_->right_->right_->augment_.is_red_ == false);
        assert(bin.root_->right_->right_->right_->value_ == 26);
        assert(bin.root_->right_->right_->right_->augment_.is_red_ == true);

        insert(bin, 15);
        assert(bin.root_->value_ == 10);
        assert(bin.root_->augment_.is_red_ == false);
        assert(bin.root_->left_->value_ == 7);
        assert(bin.root_->left_->augment_.is_red_ == true);
        assert(bin.root_->left_->left_->value_ == 3);
        assert(bin.root_->left_->left_->augment_.is_red_ == false);
        assert(bin.root_->left_->right_->value_ == 8);
        assert(bin.root_->left_->right_->augment_.is_red_ == false);
        assert(bin.root_->right_->value_ == 18);
        assert(bin.root_->right_->augment_.is_red_ == true);
        assert(bin.root_->right_->left_->value_ == 11);
        assert(bin.root_->right_->left_->augment_.is_red_ == false);
        assert(bin.root_->right_->left_->right_->value_ == 15);
        assert(bin.root_->right_->left_->right_->augment_.is_red_ == true);
        assert(bin.root_->right_->right_->value_ == 22);
        assert(bin.root_->right_->right_->augment_.is_red_ == false);
        assert(bin.root_->right_->right_->right_->value_ == 26);
        assert(bin.root_->right_->right_->right_->augment_.is_red_ == true);
        std::cout << "ok" << std::endl;
    }
}

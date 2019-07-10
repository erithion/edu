/* Additional material
*/

#include "tree_warren_stout.hpp"
#include <cassert>
#include <iostream>

int main()
{
    std::cout << "BINARY TREE WITH WARREN-STOUT BALANCING IMPLEMENTATION TESTS" << std::endl;
    // checking straighting the tree up
    {
        std::cout << "checking transformation to degenerate right list ...";
        bst::binary::tree_type<int> bin;
        insert(bin, { 5, 6, 1, 3, 8, 10, 7, 2 });
        auto sz = to_right_list(bin);
        assert(bin->value_ == 1);
        assert(bin->right_->value_ == 2);
        assert(bin->right_->right_->value_ == 3);
        assert(bin->right_->right_->right_->value_ == 5);
        assert(bin->right_->right_->right_->right_->value_ == 6);
        assert(bin->right_->right_->right_->right_->right_->value_ == 7);
        assert(bin->right_->right_->right_->right_->right_->right_->value_ == 8);
        assert(bin->right_->right_->right_->right_->right_->right_->right_->value_ == 10);
        assert(bin->right_->right_->right_->right_->right_->right_->right_->right_ == nullptr);
        assert(sz == 8);
        std::cout << "ok" << std::endl;
    }
    // balance
    {
        std::cout << "checking balancing (transforming to a complete tree) ...";
        bst::binary::tree_type<int> bin;
        insert(bin, { 2, 7, 4, 6, 3, 8, 1, 5 });
        balance(bin);
        // Now must be
        //                              5
        //                  3                       7
        //              2       4               6       8
        //          1
        assert(bin->value_ == 5);
        assert(bin->left_->value_ == 3);
        assert(bin->left_->left_->value_ == 2);
        assert(bin->left_->left_->left_->value_ == 1);
        assert(bin->left_->right_->value_ == 4);
        assert(bin->right_->value_ == 7);
        assert(bin->right_->left_->value_ == 6);
        assert(bin->right_->right_->value_ == 8);
        std::cout << "ok" << std::endl;
    }
}

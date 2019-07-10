/* Additional material
*/

#include "tree_avl.hpp"
#include <cassert>
#include <iostream>

int main()
{
    std::cout << "AVL TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking AVL size ...";
        bst::avl::tree_type<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        assert(size(bin) == 7);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for right-right insert ...";
        bst::avl::tree_type<int> bin;
        insert(bin, {11, 5, 15, 4, 7, 16, 8});
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 5);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->right_->value_ == 7);
        assert(bin->left_->right_->right_->value_ == 8);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);

        insert(bin, 9);
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 5);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->right_->value_ == 8);
        assert(bin->left_->right_->left_->value_ == 7);
        assert(bin->left_->right_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for right-left insert ...";
        bst::avl::tree_type<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 9 });
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 5);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->right_->value_ == 7);
        assert(bin->left_->right_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);

        insert(bin, 8);
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 5);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->right_->value_ == 8);
        assert(bin->left_->right_->left_->value_ == 7);
        assert(bin->left_->right_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for left-left insert ...";
        bst::avl::tree_type<int> bin;
        insert(bin, { 11, 7, 15, 5, 9, 16, 3 });
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 7);
        assert(bin->left_->left_->value_ == 5);
        assert(bin->left_->left_->left_->value_ == 3);
        assert(bin->left_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);

        insert(bin, 1);
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 7);
        assert(bin->left_->left_->value_ == 3);
        assert(bin->left_->left_->left_->value_ == 1);
        assert(bin->left_->left_->right_->value_ == 5);
        assert(bin->left_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for left-right insert ...";
        bst::avl::tree_type<int> bin;
        insert(bin, { 11, 7, 15, 5, 9, 16, 3 });
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 7);
        assert(bin->left_->left_->value_ == 5);
        assert(bin->left_->left_->left_->value_ == 3);
        assert(bin->left_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);

        insert(bin, 4);
        assert(bin->value_ == 11);
        assert(bin->left_->value_ == 7);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->left_->left_->value_ == 3);
        assert(bin->left_->left_->right_->value_ == 5);
        assert(bin->left_->right_->value_ == 9);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
}

/* Additional material
*/

#include "tree_search/tree_avl.hpp"
#include "tree_search/capability_rotate.hpp"
#include "tree_search/tree.hpp"
#include <cassert>
#include <iostream>

namespace ts = tree_search;

template <typename T>
using tree = ts::tree<T, ts::aux::avl_augment, ts::rotate_cap, ts::avl_insert_cap>;

int main()
{
    std::cout << "AVL TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking AVL size ...";
        tree<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        assert(size(bin) == 7);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for right-right insert ...";
        tree<int> bin;
        insert(bin, {11, 5, 15, 4, 7, 16, 8});
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 5);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->right_->value_ == 7);
        assert(bin.root_->left_->right_->right_->value_ == 8);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);

        insert(bin, 9);
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 5);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->right_->value_ == 8);
        assert(bin.root_->left_->right_->left_->value_ == 7);
        assert(bin.root_->left_->right_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for right-left insert ...";
        tree<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 9 });
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 5);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->right_->value_ == 7);
        assert(bin.root_->left_->right_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);

        insert(bin, 8);
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 5);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->right_->value_ == 8);
        assert(bin.root_->left_->right_->left_->value_ == 7);
        assert(bin.root_->left_->right_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for left-left insert ...";
        tree<int> bin;
        insert(bin, { 11, 7, 15, 5, 9, 16, 3 });
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 7);
        assert(bin.root_->left_->left_->value_ == 5);
        assert(bin.root_->left_->left_->left_->value_ == 3);
        assert(bin.root_->left_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);

        insert(bin, 1);
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 7);
        assert(bin.root_->left_->left_->value_ == 3);
        assert(bin.root_->left_->left_->left_->value_ == 1);
        assert(bin.root_->left_->left_->right_->value_ == 5);
        assert(bin.root_->left_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for left-right insert ...";
        tree<int> bin;
        insert(bin, { 11, 7, 15, 5, 9, 16, 3 });
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 7);
        assert(bin.root_->left_->left_->value_ == 5);
        assert(bin.root_->left_->left_->left_->value_ == 3);
        assert(bin.root_->left_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);

        insert(bin, 4);
        assert(bin.root_->value_ == 11);
        assert(bin.root_->left_->value_ == 7);
        assert(bin.root_->left_->left_->value_ == 4);
        assert(bin.root_->left_->left_->left_->value_ == 3);
        assert(bin.root_->left_->left_->right_->value_ == 5);
        assert(bin.root_->left_->right_->value_ == 9);
        assert(bin.root_->right_->value_ == 15);
        assert(bin.root_->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
}

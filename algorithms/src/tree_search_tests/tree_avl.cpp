/* Additional material
*/

#include "tree_search/tree_avl.hpp"
#include "tree_search/tree.hpp"
#include "tree_search/capability_search.hpp"
#include <cassert>
#include <iostream>
#include <vector>

namespace ts = tree_search;

template <typename T>
using tree = ts::tree<T, ts::aux::avl_augment, ts::capability_insert_avl, ts::capability_search>;

// Must fail to compile: rotation is forbidden for auto-balancing trees
//#include "tree_search/capability_rotate.hpp"
//template <typename T>
//using tree = ts::tree<T, ts::aux::avl_augment, ts::capability_rotate, ts::capability_insert_avl>;

// Must fail to compile: capability_insert_avl must go with avl_augment
//template <typename T>
//using tree = ts::tree<T, ts::aux::empty, ts::capability_rotate, ts::capability_insert_avl>;

void check_search() {

    std::cout << "checking search ...";
    tree<char> bin;
    insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
    // preorder
    {
        auto it = search(ts::tag_preorder{}, bin, [](int v) {return v == 'F' || v == 'D' || v == 'E'; });
        std::vector<int> res(begin(it), end(it));
        assert(res == std::vector<int>({ 'D', 'F', 'E' }));
    }
    // inorder
    {
        auto it = search(ts::tag_inorder{}, bin, [](int v) {return v == 'D' || v == 'E' || v == 'F'; });
        std::vector<int> res(begin(it), end(it));
        assert(res == std::vector<int>({ 'D', 'E', 'F' }));
    }
    // postorder
    {
        auto it = search(ts::tag_postorder{}, bin, [](int v) {return v == 'D' || v == 'E' || v == 'F'; });
        std::vector<int> res(begin(it), end(it));
        assert(res == std::vector<int>({ 'E', 'F', 'D' }));
    }
    std::cout << "ok" << std::endl;
}

int main()
{
    std::cout << "AVL TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking AVL size ...";
        tree<int> bin;
        insert(bin, 11); // rvalue
        insert(bin, { 5, 15, 4, 7, 16 }); // initializer_list
        int a = 8;
        insert(bin, a); // lvalue
        assert(size(bin) == 7);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for right-right insert ...";
        tree<int> bin;
        insert(bin, {11, 5, 15, 4, 7, 16, 8});
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->right_->value_ == 7);
        assert(ts::aux::access(bin)->left_->right_->right_->value_ == 8);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);

        insert(bin, 9);
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->right_->value_ == 8);
        assert(ts::aux::access(bin)->left_->right_->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->right_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for right-left insert ...";
        tree<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 9 });
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->right_->value_ == 7);
        assert(ts::aux::access(bin)->left_->right_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);

        insert(bin, 8);
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->right_->value_ == 8);
        assert(ts::aux::access(bin)->left_->right_->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->right_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for left-left insert ...";
        tree<int> bin;
        insert(bin, { 11, 7, 15, 5, 9, 16, 3 });
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->left_->left_->value_ == 3);
        assert(ts::aux::access(bin)->left_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);

        insert(bin, 1);
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->left_->value_ == 3);
        assert(ts::aux::access(bin)->left_->left_->left_->value_ == 1);
        assert(ts::aux::access(bin)->left_->left_->right_->value_ == 5);
        assert(ts::aux::access(bin)->left_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking AVL invariant for left-right insert ...";
        tree<int> bin;
        insert(bin, { 11, 7, 15, 5, 9, 16, 3 });
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->left_->left_->value_ == 3);
        assert(ts::aux::access(bin)->left_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);

        insert(bin, 4);
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->left_->left_->value_ == 3);
        assert(ts::aux::access(bin)->left_->left_->right_->value_ == 5);
        assert(ts::aux::access(bin)->left_->right_->value_ == 9);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);
        std::cout << "ok" << std::endl;
    }

    check_search();
}

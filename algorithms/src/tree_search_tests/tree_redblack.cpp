/* Additional material
*/

#include "tree_search/tree_redblack.hpp"
#include "tree_search/capability_traverse.hpp"
#include "tree_search/capability_search.hpp"
#include "tree_search/tree.hpp"
#include <cassert>
#include <iostream>
#include <vector>

namespace ts = tree_search;

template <typename T>
using tree = ts::tree<T, ts::aux::redblack_augment, ts::capability_traverse, ts::capability_insert_redblack, ts::capability_search>;

// Must fail to compile: capability_insert_redblack works only when redblack_augment type has also been specified
//template <typename T>
//using tree = ts::tree<T, ts::aux::empty, ts::capability_traverse, ts::capability_insert_redblack>;

// Must fail to compile: rotation is forbidden for auto-balancing trees
//template <typename T>
//using tree = ts::tree<T, ts::aux::redblack_augment, ts::capability_rotate, ts::capability_insert_redblack>;

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
    std::cout << "RED-BLACK TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking red-black tree size ...";
        tree<int> bin;
        insert(bin, 11); // rvalue
        insert(bin, { 5, 15, 4, 7, 16 }); // initializer_list
        int a = 8;
        insert(bin, a); // lvalue
        assert(size(bin) == 7);
        std::cout << "ok" << std::endl;
    }
    // checking inorder traversing
    {
        std::cout << "checking inorder traversing ...";
        tree<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        std::vector<int> res;
        for (auto v : traverse(ts::tag_inorder{}, bin)) res.push_back(v);
        assert(res == std::vector<int>({ 4, 5, 7, 8, 11, 15, 16 }));
        std::cout << "ok" << std::endl;
        // rotate(ts::rotate_right_tag{}, bin); // must emit compile time error for rotate_cap has not been added into the tree
    }
    //
    {
        std::cout << "checking red-black tree invariant ...";
        tree<int> bin;
        insert(bin, {11, 5, 15, 4, 7, 16, 8});
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->left_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->right_->value_ == 7);
        assert(ts::aux::access(bin)->left_->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->right_->right_->value_ == 8);
        assert(ts::aux::access(bin)->left_->right_->right_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);
        assert(ts::aux::access(bin)->right_->right_->augment_.is_red_ == true);

        insert(bin, 9);
        assert(ts::aux::access(bin)->value_ == 11);
        assert(ts::aux::access(bin)->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->value_ == 5);
        assert(ts::aux::access(bin)->left_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->left_->left_->value_ == 4);
        assert(ts::aux::access(bin)->left_->left_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->right_->value_ == 8);
        assert(ts::aux::access(bin)->left_->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->right_->right_->value_ == 9);
        assert(ts::aux::access(bin)->left_->right_->right_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->left_->right_->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->right_->left_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->right_->value_ == 16);
        assert(ts::aux::access(bin)->right_->right_->augment_.is_red_ == true);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking red-black tree insertion against MIT example covering all 3 cases ...";
        tree<int> bin;
        insert(bin, { 7, 3, 18, 10, 22, 8, 11, 26 });
        assert(ts::aux::access(bin)->value_ == 7);
        assert(ts::aux::access(bin)->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->value_ == 3);
        assert(ts::aux::access(bin)->left_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->value_ == 18);
        assert(ts::aux::access(bin)->right_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->left_->value_ == 10);
        assert(ts::aux::access(bin)->right_->left_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->left_->left_->value_ == 8);
        assert(ts::aux::access(bin)->right_->left_->left_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->left_->right_->value_ == 11);
        assert(ts::aux::access(bin)->right_->left_->right_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->right_->value_ == 22);
        assert(ts::aux::access(bin)->right_->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->right_->right_->value_ == 26);
        assert(ts::aux::access(bin)->right_->right_->right_->augment_.is_red_ == true);

        insert(bin, 15);
        assert(ts::aux::access(bin)->value_ == 10);
        assert(ts::aux::access(bin)->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->value_ == 7);
        assert(ts::aux::access(bin)->left_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->left_->left_->value_ == 3);
        assert(ts::aux::access(bin)->left_->left_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->left_->right_->value_ == 8);
        assert(ts::aux::access(bin)->left_->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->value_ == 18);
        assert(ts::aux::access(bin)->right_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->left_->value_ == 11);
        assert(ts::aux::access(bin)->right_->left_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->left_->right_->value_ == 15);
        assert(ts::aux::access(bin)->right_->left_->right_->augment_.is_red_ == true);
        assert(ts::aux::access(bin)->right_->right_->value_ == 22);
        assert(ts::aux::access(bin)->right_->right_->augment_.is_red_ == false);
        assert(ts::aux::access(bin)->right_->right_->right_->value_ == 26);
        assert(ts::aux::access(bin)->right_->right_->right_->augment_.is_red_ == true);
        std::cout << "ok" << std::endl;
    }

    check_search();
}

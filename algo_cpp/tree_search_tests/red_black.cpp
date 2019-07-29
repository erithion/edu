/* Additional material
*/

#include "tree_search/red_black.hpp"
#include <cassert>
#include <iostream>

int main()
{
    using namespace tree_search;
    std::cout << "RED-BLACK TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking red-black tree size ...";
        red_black::tree_type<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        assert(size(bin) == 7);
        std::cout << "ok" << std::endl;
    }
    // checking inorder traversing
    {
        std::cout << "checking inorder traversing ...";
        red_black::tree_type<int> bin;
        insert(bin, { 11, 5, 15, 4, 7, 16, 8 });
        std::vector<int> res;
        traverse(tree_search::red_black::inorder_tag{}, bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<int>({ 4, 5, 7, 8, 11, 15, 16 }));
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking red-black tree invariant ...";
        red_black::tree_type<int> bin;
        insert(bin, {11, 5, 15, 4, 7, 16, 8});
        assert(bin->value_ == 11);
        assert(bin->is_red_ == false);
        assert(bin->left_->value_ == 5);
        assert(bin->left_->is_red_ == true);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->left_->is_red_ == false);
        assert(bin->left_->right_->value_ == 7);
        assert(bin->left_->right_->is_red_ == false);
        assert(bin->left_->right_->right_->value_ == 8);
        assert(bin->left_->right_->right_->is_red_ == true);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->is_red_ == false);
        assert(bin->right_->right_->value_ == 16);
        assert(bin->right_->right_->is_red_ == true);

        insert(bin, 9);
        assert(bin->value_ == 11);
        assert(bin->is_red_ == false);
        assert(bin->left_->value_ == 5);
        assert(bin->left_->is_red_ == true);
        assert(bin->left_->left_->value_ == 4);
        assert(bin->left_->left_->is_red_ == false);
        assert(bin->left_->right_->value_ == 8);
        assert(bin->left_->right_->is_red_ == false);
        assert(bin->left_->right_->right_->value_ == 9);
        assert(bin->left_->right_->right_->is_red_ == true);
        assert(bin->left_->right_->left_->value_ == 7);
        assert(bin->left_->right_->left_->is_red_ == true);
        assert(bin->right_->value_ == 15);
        assert(bin->right_->is_red_ == false);
        assert(bin->right_->right_->value_ == 16);
        assert(bin->right_->right_->is_red_ == true);
        std::cout << "ok" << std::endl;
    }
    //
    {
        std::cout << "checking red-black tree insertion against MIT example covering all 3 cases ...";
        red_black::tree_type<int> bin;
        insert(bin, { 7, 3, 18, 10, 22, 8, 11, 26 });
        assert(bin->value_ == 7);
        assert(bin->is_red_ == false);
        assert(bin->left_->value_ == 3);
        assert(bin->left_->is_red_ == false);
        assert(bin->right_->value_ == 18);
        assert(bin->right_->is_red_ == true);
        assert(bin->right_->left_->value_ == 10);
        assert(bin->right_->left_->is_red_ == false);
        assert(bin->right_->left_->left_->value_ == 8);
        assert(bin->right_->left_->left_->is_red_ == true);
        assert(bin->right_->left_->right_->value_ == 11);
        assert(bin->right_->left_->right_->is_red_ == true);
        assert(bin->right_->right_->value_ == 22);
        assert(bin->right_->right_->is_red_ == false);
        assert(bin->right_->right_->right_->value_ == 26);
        assert(bin->right_->right_->right_->is_red_ == true);

        insert(bin, 15);
        assert(bin->value_ == 10);
        assert(bin->is_red_ == false);
        assert(bin->left_->value_ == 7);
        assert(bin->left_->is_red_ == true);
        assert(bin->left_->left_->value_ == 3);
        assert(bin->left_->left_->is_red_ == false);
        assert(bin->left_->right_->value_ == 8);
        assert(bin->left_->right_->is_red_ == false);
        assert(bin->right_->value_ == 18);
        assert(bin->right_->is_red_ == true);
        assert(bin->right_->left_->value_ == 11);
        assert(bin->right_->left_->is_red_ == false);
        assert(bin->right_->left_->right_->value_ == 15);
        assert(bin->right_->left_->right_->is_red_ == true);
        assert(bin->right_->right_->value_ == 22);
        assert(bin->right_->right_->is_red_ == false);
        assert(bin->right_->right_->right_->value_ == 26);
        assert(bin->right_->right_->right_->is_red_ == true);
        std::cout << "ok" << std::endl;
    }
}

/* Additional material
*/

#include "tree_search/tree.hpp"
#include "tree_search/capability_insert.hpp"
#include "tree_search/capability_traverse.hpp"
#include "tree_search/capability_rotate.hpp"
#include "tree_search/tree_avl.hpp"
#include "tree_search/tree_redblack.hpp"
#include "tree_search/tree_interval.hpp"
#include "tree_search/capability_search_interval.hpp"
#include <cassert>
#include <iostream>

namespace ts = tree_search;

template <typename T>
using tree_simple = ts::tree<T, ts::aux::empty ,ts::capability_traverse, ts::capability_insert, ts::capability_rotate>;

template <typename T>
using tree_avl = ts::tree<T, ts::aux::avl_augment, ts::capability_insert_avl>;

template <typename T>
using tree_redblack = ts::tree<T, ts::aux::redblack_augment, ts::capability_traverse, ts::capability_insert_redblack>;

template <typename T>
using tree_interval = ts::tree<std::pair< T, T>, ts::aux::interval_augment<T>, ts::capability_traverse, ts::capability_insert_interval, ts::capability_search_interval>;

int main()
{
    std::cout << "OVERLOADING RESOLUTION TESTS" << std::endl;
    // insert
    {
        std::cout << "inserting into various trees ...";
        tree_simple<int> bin_simple;
        tree_avl<int> bin_avl;
        tree_redblack<int> bin_redblack;
        tree_interval<int> bin_interval;
        // rvalues
        insert(bin_simple, 2);
        insert(bin_avl, 2);
        insert(bin_redblack, 2);
        insert(bin_interval, std::make_pair(20, 36));
        // initializer_list
        insert(bin_simple, { 7, 4, 6, 3, 8, 1 });
        insert(bin_avl, { 7, 4, 6, 3, 8, 1 });
        insert(bin_redblack, { 7, 4, 6, 3, 8, 1 });
        insert(bin_interval, 
            { std::make_pair(3, 41)
              , std::make_pair(29, 99)
              , std::make_pair(0, 1) });
        // lvalues
        int a = 5;
        auto p = std::make_pair(10, 15);
        insert(bin_simple, a);
        insert(bin_avl, a);
        insert(bin_redblack, a);
        insert(bin_interval, p);

        assert(bin_simple.root_->value_ == 2);
        assert(bin_simple.root_->left_->value_ == 1);
        assert(bin_simple.root_->right_->value_ == 7);
        assert(bin_simple.root_->right_->left_->value_ == 4);
        assert(bin_simple.root_->right_->right_->value_ == 8);
        assert(bin_simple.root_->right_->left_->left_->value_ == 3);
        assert(bin_simple.root_->right_->left_->right_->value_ == 6);
        assert(bin_simple.root_->right_->left_->right_->left_->value_ == 5);

        assert(bin_avl.root_->value_ == 4);
        assert(bin_avl.root_->left_->value_ == 2);
        assert(bin_avl.root_->right_->value_ == 7);
        assert(bin_avl.root_->left_->left_->value_ == 1);
        assert(bin_avl.root_->left_->right_->value_ == 3);
        assert(bin_avl.root_->right_->left_->value_ == 6);
        assert(bin_avl.root_->right_->right_->value_ == 8);
        assert(bin_avl.root_->right_->left_->left_->value_ == 5);

        assert(bin_redblack.root_->value_ == 4);
        assert(bin_redblack.root_->left_->value_ == 2);
        assert(bin_redblack.root_->right_->value_ == 7);
        assert(bin_redblack.root_->left_->left_->value_ == 1);
        assert(bin_redblack.root_->left_->right_->value_ == 3);
        assert(bin_redblack.root_->right_->left_->value_ == 6);
        assert(bin_redblack.root_->right_->right_->value_ == 8);
        assert(bin_redblack.root_->right_->left_->left_->value_ == 5);

        assert(bin_interval.root_->value_ == std::make_pair(20, 36));
        assert(bin_interval.root_->augment_.max_ == 99);
        assert(bin_interval.root_->left_->value_ == std::make_pair(3, 41));
        assert(bin_interval.root_->left_->augment_.max_ == 41);
        assert(bin_interval.root_->right_->value_ == std::make_pair(29, 99));
        assert(bin_interval.root_->right_->augment_.max_ == 99);
        assert(bin_interval.root_->left_->left_->value_ == std::make_pair(0, 1));
        assert(bin_interval.root_->left_->left_->augment_.max_ == 1);
        assert(bin_interval.root_->left_->right_->value_ == std::make_pair(10, 15));
        assert(bin_interval.root_->left_->right_->augment_.max_ == 15);

        search(ts::tag_inorder{}, bin_interval, std::make_pair(-8, 7), [](const auto& v) {});
        // below must fail
//        search(ts::tag_inorder{}, bin_simple, 10, [](const auto& v) {});
//        search(ts::tag_inorder{}, bin_avl, 10, [](const auto& v) {});
//        search(ts::tag_inorder{}, bin_redblack, 10, [](const auto& v) {});

        std::cout << "ok" << std::endl;
    }
}

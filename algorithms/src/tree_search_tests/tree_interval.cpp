/* Additional material
*/

#include "tree_search/tree_interval.hpp"
#include "tree_search/capability_traverse.hpp"
#include "tree_search/capability_search_interval.hpp"
#include "tree_search/tree.hpp"
#include <cassert>
#include <iostream>
#include <array>

namespace ts = tree_search;

template <typename T>
using tree = ts::tree<std::pair< T, T>, ts::aux::interval_augment<T>, ts::capability_traverse, ts::capability_insert_interval, ts::capability_search_interval>;

// Must fail to compile: capability_search_interval works only when interval_augment type has also been specified
//template <typename T>
//using tree = ts::tree<std::pair< T, T>, ts::aux::empty, ts::capability_traverse, ts::capability_insert_interval, ts::capability_search_interval>;

int main()
{
    std::cout << "INTERVAL TREE IMPLEMENTATION TESTS" << std::endl;
    //
    {
        std::cout << "checking insertion ...";
        tree<int> bin;
        insert(bin, std::make_pair(1, 10)); // rvalue
        insert(bin, { std::make_pair(-7, 3), std::make_pair(-10, 20)}); // init list
        auto vv = std::make_pair(17, 35);
        insert(bin, vv); // lvalue
        assert(bin.root_->value_ == std::make_pair(-7, 3));
        assert(bin.root_->left_->value_ == std::make_pair(-10, 20));
        assert(bin.root_->right_->value_ == std::make_pair(1, 10));
        assert(bin.root_->right_->right_->value_ == std::make_pair(17, 35));
        std::cout << "ok" << std::endl;
    }
    // checking inorder traversing. only the compilation compliance as more thorough tests are implemented elsewhere
    {
        std::cout << "checking inorder traversing ...";
        tree<int> bin;
        insert(bin, { std::make_pair(1, 10), std::make_pair(-7, 3), std::make_pair(-10, 20), std::make_pair(17, 35) });
        std::vector<tree<int>::value_type> res;
        for (auto v: traverse(ts::tag_inorder{}, bin)) res.push_back(v);
        std::vector<tree<int>::value_type> truth{ std::make_pair(-10, 20), std::make_pair(-7, 3), std::make_pair(1, 10), std::make_pair(17, 35) };
        assert(res == truth);
        std::cout << "ok" << std::endl;
    }
    // checking interval inorder search 
    {
        std::cout << "checking search (inorder) ...";
        tree<int> bin;
        insert(bin, { std::make_pair(1, 10), std::make_pair(-7, 3), std::make_pair(-10, 20), std::make_pair(17, 35) });
        std::vector<tree<int>::value_type> res;
        search(ts::tag_inorder{}, bin, std::make_pair(-8, 7), [&res](const auto& v) {
            res.push_back(v);
        });
        std::vector<tree<int>::value_type> truth{ std::make_pair(-10, 20), std::make_pair(-7, 3), std::make_pair(1, 10) };
        assert(res == truth);
        std::cout << "ok" << std::endl;
    }
    // checking interval preorder search
    {
        std::cout << "checking search (preorder) ...";
        tree<int> bin;
        insert(bin, { std::make_pair(1, 10), std::make_pair(-7, 3), std::make_pair(-10, 20), std::make_pair(17, 35) });
        std::vector<tree<int>::value_type> res;
        search(ts::tag_preorder{}, bin, std::make_pair(-8, 7), [&res](const auto& v) {
            res.push_back(v);
        });

        // Must fail to compile as search has been defined 3 predefined traversal tags only
        //search(ts::aux::empty{}, bin, std::make_pair(-8, 7), [](const auto& v) {});

        std::vector<tree<int>::value_type> truth{ std::make_pair(-7, 3), std::make_pair(-10, 20), std::make_pair(1, 10) };
        assert(res == truth);
        std::cout << "ok" << std::endl;
    }
    // checking interval postorder search
    {
        std::cout << "checking search (postorder) ...";
        tree<int> bin;
        insert(bin, { std::make_pair(1, 10), std::make_pair(-7, 3), std::make_pair(-10, 20), std::make_pair(17, 35) });
        std::vector<tree<int>::value_type> res;
        search(ts::tag_postorder{}, bin, std::make_pair(-8, 7), [&res](const auto& v) {
            res.push_back(v);
        });
        std::vector<tree<int>::value_type> truth{ std::make_pair(-10, 20), std::make_pair(1, 10), std::make_pair(-7, 3) };
        assert(res == truth);
        std::cout << "ok" << std::endl;
    }
    // checking interval search from geeks
    {
        std::cout << "checking search by geeks' example ...";
        tree<int> bin;
        insert( bin
              , { std::make_pair(15, 20)
                , std::make_pair(10, 30)
                , std::make_pair(17, 19)
                , std::make_pair(5, 20)
                , std::make_pair(12, 15)
                , std::make_pair(30, 40) });
        std::vector<tree<int>::value_type> res;
        search(ts::tag_inorder{}, bin, std::make_pair(6, 7), [&res](const auto& v) {
            res.push_back(v);
        });
        std::vector<tree<int>::value_type> truth{ std::make_pair(5, 20) };
        assert(res == truth);
        std::cout << "ok" << std::endl;
    }
    // checking interval search from wiki
    {
        std::cout << "checking search by wiki's example ...";
        tree<int> bin;
        insert(bin
            , { std::make_pair(20, 36)
              , std::make_pair(3, 41)
              , std::make_pair(29, 99)
              , std::make_pair(0, 1)
              , std::make_pair(10, 15) });
        std::vector<tree<int>::value_type> res;
        search(ts::tag_inorder{}, bin, std::make_pair(40, 60), [&res](const auto& v) {
            res.push_back(v);
        });
        std::vector<tree<int>::value_type> truth{ std::make_pair(3, 41), std::make_pair(29, 99) };
        assert(res == truth);
        std::cout << "ok" << std::endl;
    }
}

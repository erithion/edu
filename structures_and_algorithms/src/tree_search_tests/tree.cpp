/* Additional material
*/

#include "tree_search/tree.hpp"
#include "tree_search/capability_insert.hpp"
#include "tree_search/capability_traverse.hpp"
#include "tree_search/capability_rotate.hpp"
#include <cassert>
#include <iostream>

namespace ts = tree_search;

template <typename T>
using tree = ts::tree<T, ts::aux::empty ,ts::capability_traverse, ts::capability_insert, ts::capability_rotate>;

int main()
{
    std::cout << "DEFAULT TREE IMPLEMENTATION TESTS" << std::endl;
    // checking default template calls
    {
        std::cout << "checking default template calls from custom tree ...";
        tree<int> bin;
        insert(bin, 10);
        assert(bin.root_->value_ == 10);
        std::cout << "ok" << std::endl;
    }
    // checking lvalue/rvalue insert
    {
        struct A {
            A(int a) : a_(a) {
            }
            A(A&& a) {
                std::swap(this->a_, a.a_);
                return;
            }
            A(const A& a) {
                this->a_ = a.a_;
            }

            bool operator <(const A& a) const {
                return this->a_ < a.a_;
            }
            bool operator >(const A& a) const {
                return this->a_ > a.a_;
            }
            int a_;
        };

        std::cout << "checking lvalue/rvalue conformance of insertion ...";
        tree<A> bin;
        A a{ 10 };
        insert(bin, a); // lvalue
        insert(bin, A{ 20 }); // rvalue
        A tmp{ 30 };
        insert(bin, std::move(tmp)); // rvalue manual-made
        assert(tmp.a_ != 30); // value is moved
        assert(a.a_ == 10); // value is copied
        assert(bin.root_->value_.a_ == 10);
        assert(bin.root_->right_->value_.a_ == 20);
        assert(bin.root_->right_->right_->value_.a_ == 30);
        std::cout << "ok" << std::endl;
    }
    // checking size
    {
        std::cout << "checking size ...";
        tree<int> bin;
        insert(bin, { 5, 6, 1, 3, 8, 10, 7, 2 });
        assert(size(bin) == 8);
        std::cout << "ok" << std::endl;
    }
    // checking inorder traversing
    {
        std::cout << "checking inorder traversing ...";
        tree<char> bin;
        insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
        std::vector<char> res;
        traverse(ts::tag_inorder{}, bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<char>({ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I' }));
        std::cout << "ok" << std::endl;
    }
    // checking preorder traversing
    {
        std::cout << "checking preorder traversing ...";
        tree<char> bin;
        insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
        std::vector<char> res;
        traverse(ts::tag_preorder{}, bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<char>({ 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' }));
        std::cout << "ok" << std::endl;
    }
    // checking postorder traversing
    {
        std::cout << "checking postorder traversing ...";
        tree<char> bin;
        insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
        std::vector<char> res;
        traverse(ts::tag_postorder{}, bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<char>({ 'A', 'C', 'E', 'D', 'B', 'H', 'I', 'G', 'F' }));
        std::cout << "ok" << std::endl;
    }
    // checking right rotation
    {
        std::cout << "checking right rotation ...";
        tree<char> bin;
        insert(bin, { 'D', 'B', 'A', 'C', 'E' });
        rotate(bin, ts::rotate_right_tag{});
        assert(bin.root_->value_ == 'B');
        assert(bin.root_->left_->value_ == 'A');
        assert(bin.root_->right_->value_ == 'D');
        assert(bin.root_->right_->left_->value_ == 'C');
        assert(bin.root_->right_->right_->value_ == 'E');
        std::cout << "ok" << std::endl;
    }
    // checking left rotation
    {
        std::cout << "checking left rotation ...";
        tree<char> bin;
        insert(bin, { 'B', 'A', 'D', 'C', 'E' });
        rotate(bin, ts::rotate_left_tag{});
        assert(bin.root_->value_ == 'D');
        assert(bin.root_->left_->value_ == 'B');
        assert(bin.root_->left_->left_->value_ == 'A');
        assert(bin.root_->left_->right_->value_ == 'C');
        assert(bin.root_->right_->value_ == 'E');
        std::cout << "ok" << std::endl;
    }
}

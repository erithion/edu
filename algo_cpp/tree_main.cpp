/* Additional material
*/

#include "tree_binary.hpp"
#include <cassert>
#include <iostream>

int main()
{
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
        tree_binary<A> bin;
        A a{ 10 };
        insert(bin, a); // lvalue
        insert(bin, A{ 20 }); // rvalue
        A tmp{ 30 };
        insert(bin, std::move(tmp)); // rvalue manual-made
        assert(tmp.a_ != 30); // value is moved
        assert(a.a_ == 10); // value is copied
        assert(bin->value_.a_ == 10);
        assert(bin->right_->value_.a_ == 20);
        assert(bin->right_->right_->value_.a_ == 30);
        std::cout << "ok" << std::endl;
    }
    // checking size
    {
        std::cout << "checking size ...";
        tree_binary<int> bin;
        insert(bin, {5, 6, 1, 3, 8, 10, 7, 2});
        assert(size(bin) == 8);
        std::cout << "ok" << std::endl;
    }
    // checking inorder traversing
    {
        std::cout << "checking inorder traversing ...";
        tree_binary<char> bin;
        insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
        std::vector<char> res;
        traverse<char, inorder_tag>(bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<char>({ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I' }));
        std::cout << "ok" << std::endl;
    }
    // checking preorder traversing
    {
        std::cout << "checking preorder traversing ...";
        tree_binary<char> bin;
        insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
        std::vector<char> res;
        traverse<char, preorder_tag>(bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<char>({ 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' }));
        std::cout << "ok" << std::endl;
    }
    // checking postorder traversing
    {
        std::cout << "checking postorder traversing ...";
        tree_binary<char> bin;
        insert(bin, { 'F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H' });
        std::vector<char> res;
        traverse<char, postorder_tag>(bin, [&res](auto v) {
            res.push_back(v);
        });
        assert(res == std::vector<char>({ 'A', 'C', 'E', 'D', 'B', 'H', 'I', 'G', 'F' }));
        std::cout << "ok" << std::endl;
    }
    // checking straighting the tree up
    {
        std::cout << "checking transformation to degenerate right list ...";
        tree_binary<int> bin;
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
        tree_binary<int> bin;
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

/* Additional material
- https://stackoverflow.com/questions/14001676/balancing-a-bst
- Day-Warren-Stout algorithm http://penguin.ewu.edu/~trolfe/DSWpaper/
*/

#include "tree.hpp"
#include <functional>

namespace aux {

    // Helper for template functions declaring universal references
    template <typename T>
    using remove_cvref_t = std::remove_reference_t< std::remove_cv_t<T> >;

    template <typename T>
    void rotate_right(tree_binary<T>& cur) {
        if (!cur) return;
        auto left = std::move(cur->left_);
        cur->left_ = std::move(left->right_);
        left->right_ = std::move(cur);
        cur = std::move(left);
    }

    template <typename T>
    void rotate_left(tree_binary<T>& cur) {
        if (!cur) return;
        auto right = std::move(cur->right_);
        cur->right_ = std::move(right->left_);
        right->left_ = std::move(cur);
        cur = std::move(right);
    }

    struct inorder_traverse_tag {};
    struct preorder_traverse_tag {};
    struct postorder_traverse_tag {};

    template <typename T>
    void traverse(const tree_binary<T>& tree, const std::function<void(const T&)>& fn, inorder_traverse_tag) {
        if (!tree) return;
        traverse(tree->left_, fn, inorder_traverse_tag{});
        if (fn) fn(tree->value_);
        traverse(tree->right_, fn, inorder_traverse_tag{});
    }

    template <typename T>
    void traverse(const tree_binary<T>& tree, const std::function<void(const T&)>& fn, preorder_traverse_tag) {
        if (!tree) return;
        if (fn) fn(tree->value_);
        traverse(tree->left_, fn, preorder_traverse_tag{});
        traverse(tree->right_, fn, preorder_traverse_tag{});
    }

    template <typename T>
    void traverse(const tree_binary<T>& tree, const std::function<void(const T&)>& fn, postorder_traverse_tag) {
        if (!tree) return;
        traverse(tree->left_, fn, postorder_traverse_tag{});
        traverse(tree->right_, fn, postorder_traverse_tag{});
        if (fn) fn(tree->value_);
    }
}

template <typename T>
void insert(tree_binary<aux::remove_cvref_t<T>>& tree, T&& v) { // universal reference
    using ptr_type = node<aux::remove_cvref_t<T>, trees::binary>;

    if (!tree) {
        tree = std::make_unique<ptr_type>(std::forward<T>(v));
        return;
    }
    if (v < tree->value_) insert(tree->left_, std::forward<T>(v));
    else if (v > tree->value_) insert(tree->right_, std::forward<T>(v));
}

template <typename T>
void insert(tree_binary<aux::remove_cvref_t<T>>& tree, std::initializer_list<T> ls) {
    for (auto&& v: ls) insert(tree, std::move(v));
}

template <typename T>
size_t size(const tree_binary<T>& tree) {
    if (!tree) return 0;
    return 1 + size(tree->left_) + size(tree->right_);
}

using inorder_tag = aux::inorder_traverse_tag;
using preorder_tag = aux::preorder_traverse_tag;
using postorder_tag = aux::postorder_traverse_tag;

template <typename T, typename TraverseTag>
void traverse(const tree_binary<T>& tree, const std::function<void(const T&)>& fn) {
    aux::traverse(tree, fn, TraverseTag{});
}

// Transforms a tree into a right degenerate tree (or "vine" according to the Warren-Stout algorithm's notation)
template <typename T>
size_t to_right_list(tree_binary<T>& tree) {
    size_t size = 0;
    auto   cur = std::ref(tree);
    while (cur.get())
        if (cur.get()->left_) aux::rotate_right(cur.get());
        else ++size, cur = std::ref(cur.get()->right_);
    return size;
}

// Re-balance a tree to a COMPLETE tree by the Warren-Stout algorithm 
template <typename T>
void balance(tree_binary<T>& tree) {
    auto size = to_right_list(tree);
    // We want the highest FULL tree size not bigger than the current size: n_full_highest <= size
    // Geometric progression gives n_full = 2^0 + 2^1 + 2^2 + ... + 2^k = 2^(k+1)-1 = 2^height - 1 => n_full + 1 = 2^height
    // Since any FULL tree size ultimately results in powers of 2, 
    // it's enough to find out the lower (power of 2) border of the current nonfull tree size
    // log2 and floor will do the trick
    auto height = static_cast<size_t>(std::floor(std::log2(size + 1)));
    auto n_full = static_cast<size_t>(std::pow(2, height) - 1); // -1 is due to the above formula for n_full 
    // All that in order to generate a COMPLETE tree, i.e. leftovers beyond n_full must be the leftmost elements of the tree
    // So we left-rotate the leftovers first
    auto leftovers_count = size - n_full;
    auto cur = std::ref(tree);
    for (auto j = 0; j != leftovers_count; ++j) {
        aux::rotate_left(cur.get());
        cur = std::ref(cur.get()->right_);
    }
    // Now just balance
    for (auto i = n_full / 2; i != 0; i /= 2) {
        auto cur = std::ref(tree);
        for (auto j = 0; j != i; ++j) {
            aux::rotate_left(cur.get());
            cur = std::ref(cur.get()->right_);
        }
    }
}

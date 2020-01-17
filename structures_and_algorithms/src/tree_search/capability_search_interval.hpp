#pragma once 

#include "type_traverse.hpp"
#include "tree_interval.hpp"

#include <utility>

namespace tree_search {

    struct interval_search_cap {};

    namespace interval_aux {

        // TODO: add other traversal functions
        template <typename T, typename Tree>
        void search(const std::unique_ptr<Tree>& tree, const std::pair<T, T>& interval, const std::function<void(const typename Tree::value_type&)>& fn, tag_inorder&&) {
            if (!tree) return;
            if (tree->left_ && tree->left_->augment_.max_ >= interval.first)
                search(tree->left_, interval, fn, tag_inorder{});
            if (interval.first <= tree->value_.second && !(interval.second < tree->value_.first))
                if (fn) fn(tree->value_);
            if (tree->value_.first <= interval.second)
                search(tree->right_, interval, fn, tag_inorder{});
        }

/*        template <typename Tree>
        void traverse(preorder_tag, const std::unique_ptr<Tree>& tree, const std::function<void(const typename Tree::value_type)>& fn) {
            if (!tree) return;
            if (fn) fn(tree->value_);
            traverse(preorder_tag{}, tree->left_, fn);
            traverse(preorder_tag{}, tree->right_, fn);
        }

        template <typename Tree>
        void traverse(postorder_tag, const std::unique_ptr<Tree>& tree, const std::function<void(const typename Tree::value_type&)>& fn) {
            if (!tree) return;
            traverse(postorder_tag{}, tree->left_, fn);
            traverse(postorder_tag{}, tree->right_, fn);
            if (fn) fn(tree->value_);
        }*/
    }

    // TODO: enable_if for 3 predefined tags only
    template <typename Tag, typename T, typename Tree, std::enable_if_t<std::is_base_of_v<interval_search_cap, Tree>, int> = 0>
    void search(Tag&& tag, const Tree& tree, const std::pair<T, T>& interval, const std::function<void(const typename Tree::value_type&)>& fn) {
        interval_aux::search(tree.root_, interval, fn, std::move(tag));
    }
}
#pragma once 

#include <memory>
#include <vector>
#include <functional>
#include <type_traits>

namespace tree_search {
    namespace aux {

        struct empty {};

        template <typename T, typename AugmentT>
        struct node {
            using value_type = T;
            using augment_type = AugmentT;
            using node_type = node<value_type, augment_type>;
            using ptr_type = std::unique_ptr<node_type>;

            using rr_value_type = std::remove_reference_t<std::remove_cv_t<T>>;
            using rr_augment_type = std::remove_reference_t<std::remove_cv_t<AugmentT>>;
            using rr_node_type = node<rr_value_type, rr_augment_type>;

            value_type          value_;
            augment_type        augment_;
            ptr_type            left_;
            ptr_type            right_;

            node(const value_type& v)
                : value_(v), augment_{}, left_(nullptr), right_(nullptr) {}
            node(const value_type& v, const augment_type& a)
                : value_(v), augment_(a), left_(nullptr), right_(nullptr) {}

            node(value_type&& v, augment_type&& a)
                : value_(std::move(v)), augment_(std::move(a)), left_(nullptr), right_(nullptr) {}
            node(value_type&& v)
                : value_(std::move(v)), augment_{}, left_(nullptr), right_(nullptr) {}
        };

        template <typename Tree>
        size_t size(const std::unique_ptr<Tree>& tree) {
            if (!tree) return 0;
            return 1 + size(tree->left_) + size(tree->right_);
        }
    }

    template <typename T, typename AugmentT, typename ... Capabilities>//typename Traverse, typename Insert, typename Rotate>
    struct tree : public Capabilities... { //public Traverse, public Insert, public Rotate {
        using value_type = T;
        using augment_type = AugmentT;
        std::unique_ptr<aux::node<value_type, augment_type>> root_;
    };

    template <typename ... pack>
    size_t size(const tree<pack...>& tree) {
        return aux::size(tree.root_);
    }
}
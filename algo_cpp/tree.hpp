#include <memory>
#include <vector>

template <typename T, typename TreeType>
struct node
    : public TreeType {
    using value_type = T;
    using node_type = node<T, TreeType>;

    value_type                  value_;
    std::unique_ptr<node_type>  left_;
    std::unique_ptr<node_type>  right_;

    node(const value_type& v)
        : value_(v), left_(nullptr), right_(nullptr) {}
    node(value_type&& v)
        : value_(std::move(v)), left_(nullptr), right_(nullptr) {}
};

namespace trees {
    struct binary {};
    struct avl {};
    struct red_black {};
}

template <typename T>
using tree_binary = std::unique_ptr<node<T, trees::binary>>;

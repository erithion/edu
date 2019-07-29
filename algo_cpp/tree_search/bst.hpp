#include <memory>
#include <vector>
#include <functional>
#include <type_traits>

template<typename T>
struct CTTI; // compile time type inspector; just generates a compile time error with a specified type T. Use:
             //                CTTI<T> a; 
             //                CTTI<decltype(t)> b;


namespace tree_search {
    namespace aux {
        // Hiding 'node' to prevent template versions of 'insert', 'size', etc. from being discovered by ADL
        // To avoid the potential ambiguity they must be called either by qualified name or brought in to a new namespace with 'using'
        namespace base {
            struct default_ext {};

            template <typename T, typename DerivedType = default_ext>
            struct node : public DerivedType {
                using value_type = T;
                using value_typec = std::remove_reference_t<std::remove_cv_t<T>>; // helper
                using node_type = node<T, DerivedType>;
                using node_typec = node<value_typec, DerivedType>;
                using ptr_type = std::unique_ptr<node_type>;
                using ptr_typec = std::unique_ptr<node_typec>;

                value_type          value_;
                ptr_type            left_; 
                ptr_type            right_;

                node(const value_type& v)
                    : value_(v), left_(nullptr), right_(nullptr) {}

                node(value_type&& v)
                    : value_(std::move(v)), left_(nullptr), right_(nullptr) {}
            };
        }

        template <typename T, typename Tree>
        void insert(std::unique_ptr<Tree>& tree, T&& v, const std::function<void(std::unique_ptr<Tree>&)>& fn = {}) { // universal reference
            if (!tree) tree = std::make_unique<typename Tree::node_typec>(std::forward<T>(v));
            else if (v < tree->value_) insert(tree->left_, std::forward<T>(v));
            else if (v > tree->value_) insert(tree->right_, std::forward<T>(v));
            if (fn) fn(tree);
        }

        template <typename T, typename Tree>
        void insert(std::unique_ptr<Tree>& tree, std::initializer_list<T> ls) {
            for (auto&& v : ls) insert(tree, std::move(v));
        }

        template <typename Tree>
        size_t size(const std::unique_ptr<Tree>& tree) {
            if (!tree) return 0;
            return 1 + size(tree->left_) + size(tree->right_);
        }

        struct inorder_tag {};
        struct preorder_tag {};
        struct postorder_tag {};

        template <typename Tree>
        void traverse(inorder_tag, const std::unique_ptr<Tree>& tree, const std::function<void(const typename Tree::value_typec&)>& fn) {
            if (!tree) return;
            traverse(inorder_tag{}, tree->left_, fn);
            if (fn) fn(tree->value_);
            traverse(inorder_tag{}, tree->right_, fn);
        }

        template <typename Tree>
        void traverse(preorder_tag, const std::unique_ptr<Tree>& tree, const std::function<void(const typename Tree::value_typec&)>& fn) {
            if (!tree) return;
            if (fn) fn(tree->value_);
            traverse(preorder_tag{}, tree->left_, fn);
            traverse(preorder_tag{}, tree->right_, fn);
        }

        template <typename Tree>
        void traverse(postorder_tag, const std::unique_ptr<Tree>& tree, const std::function<void(const typename Tree::value_typec&)>& fn) {
            if (!tree) return;
            traverse(postorder_tag{}, tree->left_, fn);
            traverse(postorder_tag{}, tree->right_, fn);
            if (fn) fn(tree->value_);
        }

        struct left_tag {};
        struct right_tag {};

        template <typename Tree>
        void rotate(right_tag, std::unique_ptr<Tree>& cur) {
            if (!cur) return;
            auto left = std::move(cur->left_);
            cur->left_ = std::move(left->right_);
            left->right_ = std::move(cur);
            cur = std::move(left);
        }

        template <typename Tree>
        void rotate(left_tag, std::unique_ptr<Tree>& cur) {
            if (!cur) return;
            auto right = std::move(cur->right_);
            cur->right_ = std::move(right->left_);
            right->left_ = std::move(cur);
            cur = std::move(right);
        }

        template <typename T, template <typename> class NodeType>
        void remove(std::unique_ptr<NodeType<T>>& root, const std::unique_ptr<NodeType<T>>& node);

        template <typename T, template <typename> class NodeType>
        const std::unique_ptr<NodeType<T>>& find(std::unique_ptr<NodeType<T>>& root, const T& val);

        // For the user of BST
        template <typename T>
        using def = std::unique_ptr<aux::base::node<T, aux::base::default_ext>>;
    }
}
/* Additional material
   - https://cp-algorithms.com/data_structures/segment_tree.html
   - https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/
*/

#include <vector>
#include <initializer_list>
#include <type_traits>
#include <iterator>

namespace tree_other {

    template <typename T, typename Functional, typename Enable = std::conditional_t<std::is_arithmetic_v<T>, std::true_type, std::false_type>>
    struct segment; // Binary indexed tree

    template <typename T, typename F>
    struct segment<T, F, std::false_type> {
        static_assert(std::is_arithmetic_v<T>, "This Segment tree does not implement logic for any type except arithmetic");
    };

    template <typename T, typename F>
    struct segment<T, F, std::true_type> {
        using result_type = typename F::result_type;
        std::vector<T> vec_;

        // TODO: consider auto-sorting ???
        segment(std::initializer_list<T>&& ls) 
            : vec_(tree_size(ls.size())) {
            compute(this->vec_, 0, ls.begin(), ls.end());
        }

        constexpr static size_t tree_size(size_t n) {
            // Height of a binary tree required to fit all n elements: H = ceil [ log2 n ];
            // Hence the number of nodes to fit all n elements in the leafs: 
            //    S = 1 + 2 + 4 + ... + 2 ^ H = { using geometric progression: (a^0-a^(n+1))/(1-a) } = 2^(H+1) - 1 
            // Since 2 ^ (log2 n) = n except for 'ceil' function, we can get rid of calculating log2: 
            //    S = 2 ^ (H + 1) - 1 = 2 * 2 ^ H - 1 = 2 * 2 ^ (ceil [ log2 n ]) - 1 < 2 * 2 * 2 ^ log2 n = 4 * N => S < 4 * N
            return 4*n;
        }

        template <typename It>
        void static compute(std::vector<T>& out, size_t idx, It begin, It end) {
            auto d = std::distance(begin, end);
            if (d == 1) out[idx] = *begin;
            else if (d > 1) {
                size_t m = d / 2;
                compute(out, idx * 2 + 1, begin, begin + m);
                compute(out, idx * 2 + 2, begin + m, end);
                out[idx] = F()(out[idx * 2 + 1], out[idx * 2 + 2]);
            }
        }
    };

    namespace aux {
        // is s = [x0, y0] contained in f = [x1, y1)
        bool is_contained(const std::pair<size_t, size_t>& f, const std::pair<size_t, size_t>& s) {
            return f.first <= s.first && s.second < f.second;
        }
        bool is_eq(const std::pair<size_t, size_t>& f, const std::pair<size_t, size_t>& s) {
            return f.first == s.first && s.second == f.second - 1;
        }
        // TODO: identity class for all types; and use it to shorten query algorithm
        // TODO: test out of range cases
        template <typename T, typename F, typename ... pack>
        auto query(const segment<T, F, pack...>& t, size_t loc, std::pair<size_t, size_t>&& cur, std::pair<size_t, size_t>&& in) {
            if (is_eq(cur, in)) return t.vec_[loc];
            size_t m = (cur.second - cur.first) / 2;
            auto left = std::make_pair(cur.first, cur.first + m);
            auto right = std::make_pair(cur.first + m, cur.second);
            if (is_contained(left, in)) return query(t, 2 * loc + 1, std::move(left), std::move(in));
            else if (is_contained(right, in)) return query(t, 2 * loc + 2, std::move(right), std::move(in));
            else return F()(query(t, 2 * loc + 1, std::move(left), std::make_pair(in.first, left.second /*last point excluded*/ - 1))
                , query(t, 2 * loc + 2, std::move(right), std::make_pair(right.first, in.second)));
        }

        template <typename T, typename F, typename ... pack>
        auto update(segment<T, F, pack...>& t, size_t loc, std::pair<size_t, size_t>&& cur, std::pair<size_t, size_t>&& at, const T& val) {
            if (is_eq(cur, at) && at.second - at.first == 0) {
                t.vec_[loc] = val;
                return;
            }
            size_t m = (cur.second - cur.first) / 2;
            auto left = std::make_pair(cur.first, cur.first + m);
            auto right = std::make_pair(cur.first + m, cur.second);
            if (is_contained(left, at)) update(t, 2 * loc + 1, std::move(left), std::move(at), val);
            else if (is_contained(right, at)) update(t, 2 * loc + 2, std::move(right), std::move(at), val);
            else {
                update(t, 2 * loc + 1, std::move(left), std::make_pair(at.first, left.second /*last point excluded*/ - 1), val);
                update(t, 2 * loc + 2, std::move(right), std::make_pair(right.first, at.second), val);
            }
            t.vec_[loc] = F()(t.vec_[2 * loc + 1], t.vec_[2 * loc + 2]);
        }
    }


    template <typename T, typename F>
    auto query(const segment<T, F, std::true_type>& s, size_t l, size_t r) {
        return aux::query(s, 0, std::make_pair(0, s.vec_.size() / 4), std::make_pair(l, r));
    }

    template <typename T, typename F>
    void update(segment<T, F, std::true_type>& s, std::pair<size_t, size_t>&& i, T&& val) {
        aux::update(s, 0, std::make_pair(0, s.vec_.size() / 4), std::move(i), val);
    }
}
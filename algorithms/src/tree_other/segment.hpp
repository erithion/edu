/* Additional material
   - https://cp-algorithms.com/data_structures/segment_tree.html
   - https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/
*/
#pragma once

#include "identity.hpp"

#include <vector>
#include <initializer_list>
#include <iterator>

namespace tree_other {

    template <typename T, typename Functional>
    struct segment {
        using result_type = typename Functional::result_type;
        using functional_type = Functional;
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
                out[idx] = functional_type()(out[idx * 2 + 1], out[idx * 2 + 2]);
            }
        }
    };

    namespace aux {
        using ::aux::identity;

        // TODO: identity class for all types; and use it to shorten query algorithm
        // TODO: test out of range cases
        // TODO: test with min/max elements to check min/max identities
        template <typename T, typename F>
        auto query(const segment<T, F>& t, size_t loc, std::pair<size_t, size_t>&& cur, const std::pair<size_t, size_t>& in) {
            if (cur.second <= in.first || cur.first > in.second) return identity<F>().value; // exit with identity elem if we have fallen out of bounds
            if (cur.first >= in.first && cur.second < in.second || cur.second - cur.first == 1) return t.vec_[loc]; // return a value if within the range or has reached a leaf 
            size_t m = (cur.second - cur.first) / 2; // otherwise continue to split the interval into two pieces and agregate the children by F
            return F()( query(t, 2 * loc + 1, std::make_pair(cur.first, cur.first + m), in)
                      , query(t, 2 * loc + 2, std::make_pair(cur.first + m, cur.second), in));
        }

        template <typename T, typename F>
        auto update(segment<T, F>& t, size_t loc, std::pair<size_t, size_t>&& cur, const std::pair<size_t, size_t>& at, const T& val) {
            if (cur.second <= at.first || cur.first > at.second) return; // exit if we have fallen out of bounds
            if (cur.second - cur.first == 1) { // update if we have reached a final leaf
                t.vec_[loc] = val;
                return;
            }
            size_t m = (cur.second - cur.first) / 2; // otherwise continue to split the interval into two pieces
            update(t, 2 * loc + 1, std::make_pair(cur.first, cur.first + m), at, val);
            update(t, 2 * loc + 2, std::make_pair(cur.first + m, cur.second), at, val);

            t.vec_[loc] = F()(t.vec_[2 * loc + 1], t.vec_[2 * loc + 2]); // build the tree up again from the updated children
        }
    }

    template <typename T, typename F>
    auto query(const segment<T, F>& s, const std::pair<size_t, size_t>& bounds) {
        return aux::query(s, 0, std::make_pair(0, s.vec_.size() / 4), bounds);
    }

    template <typename T, typename F>
    void update(segment<T, F>& s, const std::pair<size_t, size_t>& bounds, T&& val) {
        aux::update(s, 0, std::make_pair(0, s.vec_.size() / 4), bounds, val);
    }
}
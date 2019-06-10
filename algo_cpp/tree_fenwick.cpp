#include <iostream>
#include <vector>
#include <initializer_list>
#include <cassert>
#include <string>
#include <type_traits>

template <typename T, bool Enable = std::is_arithmetic_v<T>>
struct BinaryIndexedTree;

template <typename T>
struct BinaryIndexedTree<T, false> {

    static_assert(std::is_arithmetic_v<T>, "This Fenwick tree does not implement logic for any type except arithmetic");
};

template <typename T>
struct BinaryIndexedTree<T, true> {

    template <typename ThisType> // need both const and non-const versions
	struct proxy;

	using container_type = std::vector<T>;
	using value_type = typename container_type::value_type;
	using const_reference = const proxy<const BinaryIndexedTree>;
	using reference = proxy<BinaryIndexedTree>;

	BinaryIndexedTree(std::initializer_list<value_type> ls)
		: tree_(ls.size() + 1) { 
        this->init(ls); 
    }

	const_reference operator[](size_t index) const {
        return const_reference{ *this, 1, index };
	}

    const_reference range(size_t begin_idx, size_t end_idx) const {        
        return const_reference{ *this, begin_idx, end_idx };
    }

	reference operator[](size_t index) {
		return reference{ *this, 1, index };
	}

private:

    template <typename ThisType>
	struct proxy {

		proxy& operator+=(typename value_type&& v) {
			ref_.update(to_, v);
			return *this;
		}

		operator value_type() const	{
			return ref_.get(from_, to_);
		}

		ThisType& ref_;
		size_t    from_;
        size_t    to_;
	};

	friend struct proxy<const BinaryIndexedTree>;
    friend struct proxy<BinaryIndexedTree>;

    container_type tree_;

	inline int lsb(int i) const	{
		return i & -i;
	}

	void init(std::initializer_list<value_type> ls)	{
		size_t idx = 1; // due to index arithmetic we start from 1
		for (auto it = std::begin(ls); it != std::end(ls); ++idx, ++it)	{
			this->tree_[idx] += *it; // consider the current value from ls
			auto upper_idx = idx + this->lsb(idx);
			if (upper_idx <= ls.size())
				this->tree_[upper_idx] += this->tree_[idx]; // add current value to the responsible for us upper value
		}
	}

	void update(size_t idx, value_type v) {
        /* updating from the current and all responsible upwards. graphically:
                    |
                    |
                    |
                    |
                |
                |
            |
        */
		for (; idx <= tree_.size(); idx += lsb(idx))
			tree_[idx] += v;
	}

	value_type get(size_t from, size_t to) const {
        /* summing from the current and all responsible downwards. graphically:
            |
                |
                |
                    |
                    |
                    |
                    |
        */
        value_type res_to{}, res_from{};
		for (; to != 0; to -= this->lsb(to))
			res_to += this->tree_[to];
        for (--from; from != 0; from -= this->lsb(from))
            res_from += this->tree_[from];
        return res_to - res_from;
	}
};

int main()
{
    // testing creation and getters
	// 1 2 3 4 5  -> 1 3 6 10 15
	// 1 2 3 4 5     1 2 3  4  5
	{
		const BinaryIndexedTree<int> bit{ 1, 2, 3, 4, 5 };
		assert(bit[1] == 1);
		assert(bit[2] == 3);
		assert(bit[3] == 6);
		assert(bit[4] == 10);
		assert(bit[5] == 15);
	}
    // testing updates
	// 1 2 3 4 5 +-> 10 -> 1 3 16 20 25
	// 1 2 3 4 5 +->  3 -> 1 2 3  4  5
	{
		BinaryIndexedTree<int> bit{ 1, 2, 3, 4, 5 };
		bit[3] += 10;
		assert(bit[1] == 1);
		assert(bit[2] == 3);
		assert(bit[3] == 16);
		assert(bit[4] == 20);
		assert(bit[5] == 25);
	}
    // anything except arithmetic types are not supported
    {
//       BinaryIndexedTree<std::string> bit{ "1232", "sd", "fd" };
    }
    // testing the range case
    {
        BinaryIndexedTree<int> bit{ 1, 2, 3, 4, 5 };
        assert(bit.range(4, 5) == 9);
    }
    // testing the range edge cases
    {
        BinaryIndexedTree<int> bit{ 1, 2, 3, 4, 5 };
        assert(bit.range(1, 1) == 1);
        assert(bit.range(5, 5) == 5);
    }
}

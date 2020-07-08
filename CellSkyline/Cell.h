#pragma once
#include <ostream>


template<int D>
class Cell
{
public:
	unsigned short int indices[D];
	Cell(std::initializer_list<unsigned short int> il);

	unsigned short int& operator[](int i) { return indices[i]; }
	
	friend std::ostream& operator<< (std::ostream& os, const Cell& p)
	{
		os << "[";
		std::copy(std::begin(p.indices), std::end(p.indices), std::ostream_iterator<unsigned short int>(os, " "));
		os << "]";
		return os;
	}
	
	friend bool operator== (const Cell& kc1, const Cell& kc2)
	{
		return std::equal(std::begin(kc1.indices), std::end(kc1.indices), std::begin(kc2.indices));
	}

	friend bool operator!= (const Cell& kc1, const Cell& kc2)
	{
		return !(kc1 == kc2);
	}
};

template <int D>
Cell<D>::Cell(std::initializer_list<unsigned short int> il) 
{
	int i = 0;
	for (auto elem : il)
	{
		indices[i++] = elem;
	}
}


// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_SINGLETON_HPP
#define MCL_SINGLETON_HPP 1

namespace mcl
{

template <typename T>
class Singleton
{
protected:
	Singleton() {}
	virtual ~Singleton() {}
public:
	Singleton(Singleton const &) = delete;
	Singleton operator=(Singleton const &) = delete;
	static T& get()
	{
		static T instance;
		return instance;
	}
};

} // ns mcl

#endif

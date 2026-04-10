#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <format>

using std::string;

constexpr void print(const std::string_view str_fmt, auto &&...args)
{
    fputs(std::vformat(str_fmt, std::make_format_args(args...)).c_str(), stdout);
}

template <typename T>
class container
{
    std::vector<T> things{};

public:
    container()
    {
        print("default constructor\n");
    }

    container(std::initializer_list<T> il) : things{il.begin(), il.end()}
    {
        print("initilaizer-list constructor\n");
    }

    container(const container &rhs) : things{rhs.things} // copy constructor
    {
        print("copy constructor\n");
    }

    container(container &&rhs) noexcept : things{std::move(rhs.things)} // move constructor
    {
        print("move constructor\n");
    }

    ~container()
    {
        print("descructor\n");
    }

    void reset()
    {
        things.clear();
    }

    container<T> &operator=(container rhs) // copy/swap
    {
        print("copy/swap operator\n");
        std::swap(things, rhs.things);
        return *this;
    }

    std::string str() const
    {
        std::string out{};
        if (things.empty())
            return "[empty]";
        for (auto i : things)
        {
            if (out.size())
                out += ':';
            out += i;
        }
        return out;
    }
};

template<typename T>
container<T> f(container<T> o) {
    return o;
}

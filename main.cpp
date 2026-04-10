#include <iostream>
#include <memory>
#include <thread>
#include <string>
#include <concepts>
#include <format>
#include <functional>
#include <cstdlib>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/framework/features.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>

#include <boost/log/trivial.hpp>

#include <boost/json.hpp>
#include <boost/asio.hpp>

#include "my_class.hpp"
#include "container.hpp"
#include "printer.hpp"

using namespace boost::json;

template <typename T>
concept Numeric = requires(T a) {
    a + 1;
    a * 1;
};

auto arg42(const Numeric auto &arg)
{
    return arg + 42;
}

int main(int, char **)
{   
    BOOST_LOG_TRIVIAL(trace) << "A trace severity message";
    BOOST_LOG_TRIVIAL(debug) << "A debug severity message";
    BOOST_LOG_TRIVIAL(info) << "An informational severity message";
    BOOST_LOG_TRIVIAL(warning) << "A warning severity message";
    BOOST_LOG_TRIVIAL(error) << "An error severity message";
    BOOST_LOG_TRIVIAL(fatal) << "A fatal severity message";

    setenv("MY_VAR", "martin", 1);
    std::cout << getenv("MY_VAR") << std::endl;

    std::cout << "Main start..." << std::endl;
    boost::asio::io_context io;
    printer p(io);
    std::thread t{[&]{io.run();}};
    io.run();
    t.join();

    value jv = {
        {"pi", 3.141},
        {"happy", true},
        {"name", "Boost"},
        {"nothing", nullptr},
        {"answer", {{"everything", 42}}},
        {"list", {1, 0, 2}},
        {"object", {{"currency", "USD"}, {"value", 42.99}}}};

    std::string s = serialize(jv);

    std::cout << s << std::endl;

    auto n = 7.5;
    print("The answer {}\n", arg42(n));
    std::string result = std::format("My name is {} and I am {} years old", "Martin", 46);
    std::cout << result << std::endl;

    std::cout << "Hello, from cpp-playground!\n";
    auto ptr = std::make_unique<int[]>(7);
    auto ptr2 = std::make_unique<double[]>(2);

    N::my_class c1;

    c1.do_something();

    ptr2[0] = 234.34;
    ptr2[1] = 1.4;

    for (int i = 0; i < 7; i++)
    {
        ptr[i] = i;
    }

    for (int i = 0; i < 7; i++)
    {
        std::cout << ptr[i] << std::endl;
    }

    for (int i = 0; i < 2; i++)
    {
        std::cout << " " << ptr2[i] << std::endl;
    }

    container<std::string> a{"one", "two", "three", "four", "five"};
    container<std::string> b{"six", "seven", "eight", "nine", "ten"};

    std::cout << a.str() << std::endl;
    std::cout << b.str() << std::endl;

    b = a;

    std::cout << a.str() << std::endl;
    std::cout << b.str() << std::endl;

    using namespace boost::accumulators;
    {
        accumulator_set<double, features<tag::mean, tag::moment<2>, stats<tag::max>, tag::min>> acc;
        acc(34.5);
        acc(3.5);
        acc(77.3);
        acc(3.4);
        acc(4.3);
        acc(23.3);
        acc(100.5);

        std::cout << "Mean: " << mean(acc) << std::endl;
        std::cout << "Moment: " << moment<2>(acc) << std::endl;
        std::cout << "Count: " << count(acc) << std::endl;
        std::cout << "Max: " << max(acc) << std::endl;
        std::cout << "Min: " << min(acc) << std::endl;
    }

    // container c(std::move(a));
    // container<string> c{};
    // c = std::move(a);

    // std::cout << a.str() << std::endl;
    // std::cout << c.str() << std::endl;

    // f(a);

    // std::cout << *(ptr.get()) << std::endl;

    std::vector<int> nums{1, 45, 3, 9, 33, 1943};

    std::sort(nums.begin(), nums.end(), std::greater());
    for (auto &&v : nums)
    {
        std::cout << v << " ";
    }

    std::shared_ptr<int[]> shared(new int[10]());

    shared[0] = 5;
    shared[2] = 6;

    std::cout << "TEST: " << shared[0] << std::endl;
}

#pragma once
#include <boost/asio.hpp>

class printer
{
private:
    boost::asio::strand<boost::asio::io_context::executor_type> strand_;
    boost::asio::steady_timer timer1_;
    boost::asio::steady_timer timer2_;
    int count_;
public:
    printer(boost::asio::io_context& io);
    ~printer();
    void print1();
    void print2();
};

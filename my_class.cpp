#include "my_class.hpp"
#include <iostream>

using namespace N;
using namespace std;

void my_class::do_something() {
    cout << "Doing something!" << endl;
}

void my_class::close() {
    cout << "close resource" << endl;
}
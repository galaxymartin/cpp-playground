#pragma once
#include "turtle.hpp"

class Painter
{
    Turtle *turtle;

public:
    Painter(Turtle *turtle) : turtle(turtle) {}
    bool DrawCircle(int, int, int)
    {
        turtle->PenDown();
        turtle->GoTo(4,5);
        return true;
    }
};

#pragma once

#include <gmock/gmock.h>
#include "worklist_adapter.hpp"

using namespace std;

class MockWorklistAdapter : public WorklistAdapter
{
public:
    MOCK_METHOD1(getWorklist, string(int id));
};

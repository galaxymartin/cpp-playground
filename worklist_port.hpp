#pragma once
#include "worklist_adapter.hpp"

using namespace std;

class WorklistPort
{
    WorklistAdapter *worklistAdapter;

public:
    WorklistPort(WorklistAdapter *worklistAdapter) : worklistAdapter(worklistAdapter) {}
    string getWorklistOrdered(int id)
    {
        return worklistAdapter->getWorklist(id);
    }
};

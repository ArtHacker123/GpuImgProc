#pragma once

#include <CL/cl.hpp>

namespace Ocl
{

struct HoughData
{
    cl_int rho;
    cl_int angle;
    cl_int strength;
};

struct OptFlowData
{
    cl_int x;
    cl_int y;
    float u;
    float v;
};

};

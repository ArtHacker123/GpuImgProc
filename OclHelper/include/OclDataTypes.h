#pragma once

namespace Ocl
{

struct Pos
{
    int x;
    int y;
};

struct HoughData
{
    int rho;
    int angle;
    int strength;
};

struct OptFlowData
{
    int x;
    int y;
    float u;
    float v;
};

};

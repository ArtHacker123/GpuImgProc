#include "OclUtils.h"

size_t Ocl::localGroupSize(size_t size)
{
    if ((size%16) == 0)
    {
        return 16;
    }

    if ((size%8) == 0)
    {
        return 8;
    }

    return 0;
}
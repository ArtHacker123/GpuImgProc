#pragma once

#include <CL/cl.hpp>

#define OCL_PROGRAM_SOURCE(s) #s

class ExtractCoords
{
public:
    ExtractCoords(cl::Context& ctxt, cl::CommandQueue& q)
        :context(ctxt),
         queue(q)
    {
        const char sSource[] = OCL_PROGRAM_SOURCE(
            inline void cross_coords(int i, int2 pos, int2 dim, __global float2* coord, float2 uv)
            {
                float r = length(uv);
                coord[i] = (float2)( (float)(pos.x)/(float)dim.x, (float)(pos.y)/(float)dim.y );
                coord[i+1] = coord[i]+uv;
                coord[i+2] = coord[i+1];
                coord[i+3] = coord[i+1]-(float2)( (4.0*(uv.x+uv.y)*cos(M_PI_F/4.0))/((float)dim.x*r), (4.0*(uv.y-uv.x)*cos(M_PI_F/4.0))/((float)dim.y*r) );
                coord[i+4] = coord[i+1];
                coord[i+5] = coord[i+1]-(float2)( (4.0*(uv.x-uv.y)*cos(M_PI_F/4.0))/((float)dim.x*r), (4.0*(uv.y+uv.x)*cos(M_PI_F/4.0))/((float)dim.y*r) );
            }

            kernel void find_coords(read_only image2d_t uvImg, global float2* coord, int max_size)
            {
                int i = get_global_id(0);
                int index = 6*i;
                int blk_width = get_image_width(uvImg)/4;
                int2 pos1 = (int2) ( (4*(i%blk_width))+2, (4*(i/blk_width))+2 );
                int2 pos = (int2)((pos1.x*2)-get_image_width(uvImg), get_image_height(uvImg)-(pos1.y*2));
                float2 uv = read_imagef(uvImg, pos1).xy;
                float r = length(uv);
                if (r < 5.0)
                {
                    coord[index] = (float2)(0.0, 0.0);
                    coord[index+1] = coord[index];
                    coord[index+2] = coord[index];
                    coord[index+3] = coord[index];
                    coord[index+4] = coord[index];
                    coord[index+5] = coord[index];
                }
                else
                {
                    uv = (float2)(uv.x/get_image_width(uvImg), uv.y/get_image_height(uvImg));
                    cross_coords(index, pos, get_image_dim(uvImg), coord, uv);
                }
            };
        );

        cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
        mPgm = cl::Program(context, source);
        cl_int err = mPgm.build();

        kernel = cl::Kernel(mPgm, "find_coords");
    }

    ~ExtractCoords()
    {
    }

public:
    void do_work(const cl::ImageGL& uvImg, cl::BufferGL& coords)
    {
        size_t width = 0, height = 0;
        uvImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
        uvImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

        kernel.setArg(0, uvImg);
        kernel.setArg(1, coords);
        kernel.setArg(2, (int)(width*height/4));

        const size_t lsize = 1;
        const size_t wsize = (width/4)*(height/4);

        std::vector<cl::Memory> gl_objs = { uvImg, coords };
        queue.enqueueAcquireGLObjects(&gl_objs);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(wsize), cl::NDRange(lsize), NULL, &event);
        event.wait();
        queue.enqueueReleaseGLObjects(&gl_objs);
    }

private:
    cl::Program mPgm;
    cl::Kernel kernel;
    cl::Context& context;
    cl::CommandQueue& queue;
};

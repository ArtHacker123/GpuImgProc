#pragma once

#include <list>
#include "glHelper.h"

class AbsStage;

class GlPipe
{
public:
	GlPipe(GlFrameBuffer& fb);
	~GlPipe();

	void add(AbsStage* pStage);

	void process();

protected:
	GlFrameBuffer& mFrameBuf;
	std::list<AbsStage*> mPipe;
};

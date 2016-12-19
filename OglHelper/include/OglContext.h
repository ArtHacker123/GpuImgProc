#pragma once

#include "stdafx.h"

#include "glew.h"
#include <GL/GL.h>

namespace Ogl
{

class WinGlContext
{
    friend class UseWinGlContext;
public:
    WinGlContext(CWnd* pWnd) :m_pWnd(pWnd) { mHglRc = wglCreateContext(m_pWnd->GetDC()->m_hDC); };
    ~WinGlContext() { wglDeleteContext(mHglRc); };

private:
    HGLRC mHglRc;
    CWnd* m_pWnd;
};

class UseWinGlContext
{
public:
    UseWinGlContext(WinGlContext& ctxt) :mCtxt(ctxt) { wglMakeCurrent(mCtxt.m_pWnd->GetDC()->m_hDC, mCtxt.mHglRc); };
    ~UseWinGlContext() { /*SwapBuffers(mCtxt.m_pWnd->GetDC()->m_hDC);*/ wglMakeCurrent(0, 0); };

private:
    WinGlContext& mCtxt;
};

};

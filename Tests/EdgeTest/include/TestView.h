
// TestView.h : interface of the TestView class
//


#pragma once

#include <vector>
#include <xmemory>

#include "OglContext.h"
#include "GlView.h"

// TestView 

class TestView : public CWnd
{
// Construction
public:
    TestView(int w, int h, const char* fpath);
    virtual ~TestView();

// Attributes
public:

protected:
    // Overrides
    virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

    // Generated message map functions
    afx_msg void OnPaint();
    afx_msg void OnDestroy();
    afx_msg void OnTimer(UINT_PTR nIDEvent);
    afx_msg void OnSize(UINT nType, int cx, int cy);
    afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
    afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
    DECLARE_MESSAGE_MAP()

private:
    void initGL();
    void createOpenGLContext();

private:
    FILE* mFile;
    size_t mWidth;
    size_t mHeight;
    size_t mFrameSize;

    std::vector<BYTE> mYuvData;
    std::unique_ptr<GlView> mGlView;
    std::unique_ptr<Ogl::WinGlContext> mGlCtxt;
};

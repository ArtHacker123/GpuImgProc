
// Edge.h : main header file for the Edge application
//
#pragma once

#ifndef __AFXWIN_H__
    #error "include 'stdafx.h' before including this file for PCH"
#endif


// TestApp:
// See TestApp.cpp for the implementation of this class
//

class TestApp : public CWinApp
{
public:
    TestApp();


// Overrides
public:
    virtual BOOL InitInstance();
    virtual int ExitInstance();

// Implementation

public:
    DECLARE_MESSAGE_MAP()
};

extern TestApp theApp;

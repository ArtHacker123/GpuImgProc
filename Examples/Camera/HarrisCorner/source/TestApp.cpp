
// Edge.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "afxwinappex.h"
#include "afxdialogex.h"
#include "TestApp.h"
#include "TestView.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// OptFlowApp

BEGIN_MESSAGE_MAP(TestApp, CWinApp)
END_MESSAGE_MAP()


// OptFlowApp construction

TestApp::TestApp()
{
    // support Restart Manager
    m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;
#ifdef _MANAGED
    // If the application is built using Common Language Runtime support (/clr):
    //     1) This additional setting is needed for Restart Manager support to work properly.
    //     2) In your project, you must add a reference to System.Windows.Forms in order to build.
    System::Windows::Forms::Application::SetUnhandledExceptionMode(System::Windows::Forms::UnhandledExceptionMode::ThrowException);
#endif

    // TODO: replace application ID string below with unique ID string; recommended
    // format for string is CompanyName.ProductName.SubProduct.VersionInformation
    SetAppID(_T("Edge.AppID.NoVersion"));

    // TODO: add construction code here,
    // Place all significant initialization in InitInstance
}

// The one and only OptFlowApp object

TestApp theApp;


// OptFlowApp initialization

BOOL TestApp::InitInstance()
{
    // InitCommonControlsEx() is required on Windows XP if an application
    // manifest specifies use of ComCtl32.dll version 6 or later to enable
    // visual styles.  Otherwise, any window creation will fail.
    INITCOMMONCONTROLSEX InitCtrls;
    InitCtrls.dwSize = sizeof(InitCtrls);
    // Set this to include all the common control classes you want to use
    // in your application.
    InitCtrls.dwICC = ICC_WIN95_CLASSES;
    InitCommonControlsEx(&InitCtrls);

    CWinApp::InitInstance();

    // AfxInitRichEdit2() is required to use RichEdit control	
    // AfxInitRichEdit2();

    // Standard initialization
    // If you are not using these features and wish to reduce the size
    // of your final executable, you should remove from the following
    // the specific initialization routines you do not need
    // Change the registry key under which our settings are stored
    // TODO: You should modify this string to be something appropriate
    // such as the name of your company or organization
    SetRegistryKey(_T("Local AppWizard-Generated Applications"));

    // To create the main window, this code creates a new frame window
    // object and then sets it as the application's main window object
    int width = 640;
    int height = 480;

    CRect rect;
    GetClientRect(GetDesktopWindow(), &rect);
    int xpos = (rect.Width()-width) >> 1;
    int ypos = (rect.Height()-height) >> 1;
    m_pMainWnd = new TestView;
    LPCTSTR cs = AfxRegisterWndClass(CS_CLASSDC|CS_DBLCLKS, ::LoadCursor(NULL, IDC_ARROW), 0, 0);
    m_pMainWnd->CreateEx(0, cs, _T("Corner Detection (OpenCL)"), WS_OVERLAPPEDWINDOW, xpos, ypos, width, height, 0, 0);
    m_pMainWnd->ShowWindow(SW_SHOW);
    m_pMainWnd->UpdateWindow();

    return TRUE;
}

int TestApp::ExitInstance()
{
    //TODO: handle additional resources you may have added
    AfxOleTerm(FALSE);

    return CWinApp::ExitInstance();
}

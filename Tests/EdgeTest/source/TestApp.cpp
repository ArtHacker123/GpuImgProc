
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


// TestApp

BEGIN_MESSAGE_MAP(TestApp, CWinApp)
	ON_COMMAND(ID_APP_ABOUT, &TestApp::OnAppAbout)
END_MESSAGE_MAP()


// TestApp construction

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

// The one and only TestApp object

TestApp theApp;

int toInteger(const CString& data, int& i)
{
    CString temp = _T("");
    while (i < data.GetLength() && data.GetAt(i) != ' ')
    {
        temp.AppendChar(data.GetAt(i));
        ++i;
    }
    return _wtoi(temp);
}

void GetFilePath(const CString& data, int& i, CStringA& fpath)
{
    fpath = _T("");
    while (i < data.GetLength() && data.GetAt(i) != ' ')
    {
        fpath.AppendChar(data.GetAt(i));
        ++i;
    }
}

BOOL IsFilePathValid(const CStringA& fpath)
{
    FILE* file;
    BOOL status;
    fopen_s(&file, fpath, "rb");
    if (file == 0)
    {
        status = FALSE;
    }
    else
    {
        status = TRUE;
        fclose(file);
    }
    return status;
}

BOOL ProcessCmdLine(const CString& cmdLine, CStringA& fpath, int& width, int &height)
{
    int i = 0;

    width = 0;
    height = 0;
    fpath = _T("");

    while (i < cmdLine.GetLength())
    {
        wchar_t ch = cmdLine.GetAt(i);
        if (ch == '-')
        {
            ch = cmdLine.GetAt(i+1);
            i += ((cmdLine.GetAt(i+2) == ' ')?3:2);
            switch (ch)
            {
            case 'f':
                GetFilePath(cmdLine, i, fpath);
                break;
            case 'w':
                width = toInteger(cmdLine, i);
                break;
            case 'h':
                height = toInteger(cmdLine, i);
                break;
            }
        }
        else
        {
            ++i;
        }
    }

    return !(fpath == _T("") || width == 0 || height == 0);
}

// TestApp initialization

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


	// Initialize OLE libraries
	if (!AfxOleInit())
	{
		AfxMessageBox(IDP_OLE_INIT_FAILED);
		return FALSE;
	}

	AfxEnableControlContainer();

	EnableTaskbarInteraction(FALSE);

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

	int xpos = 100, ypos = 100;
    int width = 0, height = 0;

    CStringA fpath;
    BOOL status = ProcessCmdLine(m_lpCmdLine, fpath, width, height);
    BOOL fileStatus = (status == TRUE) && IsFilePathValid(fpath);
    if (status == FALSE || fileStatus == FALSE)
    {
        CString msg = _T("Command Line Usage:- -f FILE_PATH -w WIDTH -h HEIGHT\n\n\nExample:- EdgeTest.exe -f test.yuv -w 1280 -h 720");
        if (status && (fileStatus == FALSE))
        {
            msg = _T("Invalid Path: ") + CString(fpath) + _T("\n\n") + msg;
        }
        AfxMessageBox(msg);
        return FALSE;
    }

    m_pMainWnd = new TestView(width, height, fpath);
	LPCTSTR cs = AfxRegisterWndClass(CS_CLASSDC | CS_DBLCLKS, ::LoadCursor(NULL, IDC_ARROW), 0, 0);
	m_pMainWnd->CreateEx(0, cs, CString("Canny Edge Detection - "+fpath), WS_OVERLAPPEDWINDOW, xpos, ypos, width, height, 0, 0);
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

// TestApp message handlers


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// App command to run the dialog
void TestApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

// TestApp message handlers




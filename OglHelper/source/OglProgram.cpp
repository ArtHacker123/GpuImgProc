#include <afxwin.h>
#include "OglProgram.h"

using namespace Ogl;

Program::Program(const GLchar* pVs, GLint vsLength, const GLchar* pFs, GLint fsLength)
    :mProgram(0),
    mVertShader(0),
    mFragShader(0)
{
    init(pVs, vsLength, pFs, fsLength);
}

Program::~Program()
{
    cleanup();
}

void Program::init(const GLchar* pVs, GLint vsLength, const GLchar* pFs, GLint fsLength)
{
    mProgram = glCreateProgram();
    mVertShader = glCreateShader(GL_VERTEX_SHADER);
    mFragShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(mVertShader, 1, (const GLchar**)&pVs, &vsLength);
    compileShader(mVertShader);

    glShaderSource(mFragShader, 1, (const GLchar**)&pFs, &fsLength);
    compileShader(mFragShader);

    glAttachShader(mProgram, mVertShader);
    glAttachShader(mProgram, mFragShader);

    glLinkProgram(mProgram);
}

void Program::cleanup()
{
    glDetachShader(mProgram, mFragShader);
    glDetachShader(mProgram, mVertShader);

    glDeleteShader(mVertShader);
    glDeleteShader(mFragShader);

    glDeleteProgram(mProgram);
}

void Program::compileShader(GLuint shader)
{
    GLint status = 0;
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint size = 0;
        GLchar shader_status[1000];
        glGetShaderInfoLog(shader, 1000, &size, shader_status);
        AfxMessageBox(CString("Shader Failed ") + shader_status);
        exit(-1);
    }
}

void Program::setUniform1i(const GLchar* name, GLint value)
{
    GLint loc = glGetUniformLocation(mProgram, name);
    glUniform1i(loc, value);
}

void Program::setUniform1f(const GLchar* name, GLfloat value)
{
    GLint loc = glGetUniformLocation(mProgram, name);
    glUniform1f(loc, value);
}

void Program::setUniform4f(const GLchar* name, GLfloat value1, GLfloat value2, GLfloat value3, GLfloat value4)
{
    GLint loc = glGetUniformLocation(mProgram, name);
    glUniform4f(loc, value1, value2, value3, value4);
}

void Program::setUniformMatrix4fv(const GLchar* name, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    GLint loc = glGetUniformLocation(mProgram, name);
    glUniformMatrix4fv(loc, count, transpose, value);
}

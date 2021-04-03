#ifndef LIBRENDER_OFFSCREEN_H
#define LIBRENDER_OFFSCREEN_H

#include <GL/glew.h>
#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include <GL/glut.h>

class OffscreenGL {

public:
  OffscreenGL(int maxHeight, int maxWidth);
  ~OffscreenGL();

private:
  static int glutWin;
  static bool glutInitialized;
  GLuint fb;
  GLuint renderTex;
  GLuint depthTex;
};


void renderDepthMesh(int *FM, int fNum, double *VM, int vNum, double *intrinsics, int *imgSizeV, double *zNearFarV, float *depthBuffer);

#endif

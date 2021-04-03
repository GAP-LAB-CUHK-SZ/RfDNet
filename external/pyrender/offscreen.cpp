#include "offscreen.h"

int OffscreenGL::glutWin = -1;
bool OffscreenGL::glutInitialized = false;

OffscreenGL::OffscreenGL(int maxHeight, int maxWidth) {

  if (!glutInitialized) {
    int argc = 1;
    char *argv = "test";
    glutInit(&argc, &argv);
    glutInitialized = true;
  }

  glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(maxWidth, maxHeight);

  // create or set window & off-screen framebuffer
  if (glutWin < 0) {

    glutWin = glutCreateWindow("OpenGL");
    glutHideWindow();
    glewInit();
    glGenFramebuffersEXT(1, &fb);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
    glGenTextures(1, &renderTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, renderTex);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, maxWidth, maxHeight,
            0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, depthTex);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH24_STENCIL8, maxWidth, maxHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);

    glGenFramebuffersEXT(1, &fb);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, renderTex, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_ARB, depthTex, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT|GL_DEPTH_ATTACHMENT_EXT);
  } else {
    glutSetWindow(glutWin);
  }
}

OffscreenGL::~OffscreenGL() {
}

void cameraSetup(double zNear, double zFar, double *intrinsics, unsigned int imgHeight, unsigned int imgWidth) {

  double viewMat[] = {1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
  double fcv[] = {intrinsics[0], intrinsics[1]};
  double ccv[] = {intrinsics[2], intrinsics[3]};

  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_TEXTURE_2D);

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixd(viewMat);

  double left = - ccv[0] / fcv[0] * zNear;
  double bottom = (ccv[1] - (double)(imgHeight-1)) / fcv[1] * zNear;
  double right = ((double)imgWidth - 1.0 - ccv[0]) / fcv[0] * zNear;
  double top = ccv[1] / fcv[1] * zNear;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(left, right, bottom, top, zNear, zFar);
  glViewport(0, 0, imgWidth, imgHeight);
}

GLuint createDisplayList(int *fM, int fNum, double *vM, int vNum) {

  GLuint theShape;
  int *fp;
  int vIndex;

  theShape = glGenLists(1);

  glNewList(theShape, GL_COMPILE);

  glBegin(GL_TRIANGLES);
  for (int i = 0; i < fNum; i++) {
    fp = fM + i * 3;

    vIndex = fp[0] * 3;
    glVertex3d(vM[vIndex], vM[vIndex + 1], vM[vIndex + 2]);

    vIndex = fp[1] * 3;
    glVertex3d(vM[vIndex], vM[vIndex + 1], vM[vIndex + 2]);

    vIndex = fp[2] * 3;
    glVertex3d(vM[vIndex], vM[vIndex + 1], vM[vIndex + 2]);
  }
  glEnd();
  glEndList();

  return theShape;
}

void drawPatchToDepthBuffer(GLuint listName, float *depthBuffer, unsigned int imgHeight, unsigned int imgWidth, double *zNearFarV) {

  glCallList(listName);
  glFlush();

  // bug fix for Nvidia
  unsigned int paddedWidth = imgWidth % 4;
  if (paddedWidth != 0) paddedWidth = 4 - paddedWidth + imgWidth;
  else                  paddedWidth = imgWidth;

  // Read off of the depth buffer
  float *dataBuffer_depth = (float *)malloc(paddedWidth * imgHeight * sizeof(GL_FLOAT));
  glReadPixels(0, 0, paddedWidth, imgHeight, GL_DEPTH_COMPONENT, GL_FLOAT, dataBuffer_depth);

  float n = zNearFarV[0];
  float f = zNearFarV[1];
  int out_idx = 0;
  for (unsigned int i = 0; i < imgHeight; i++) {
    for (unsigned int j = 0; j < imgWidth; j++) {
      int ogl_idx = (j + (imgHeight-1-i) * paddedWidth);
      float depth = dataBuffer_depth[ogl_idx];

      if(depth < 1) {
        depthBuffer[out_idx] = -f*n/(depth*(f-n)-f);
      }
      else {
        depthBuffer[out_idx] = -1;
      }

      out_idx++;
    }
  }

  free(dataBuffer_depth);
}

void renderDepthMesh(int *FM, int fNum, double *VM, int vNum, double *intrinsics, int *imgSizeV, double *zNearFarV, float *depthBuffer) {
  OffscreenGL offscreenGL(imgSizeV[0], imgSizeV[1]);
  cameraSetup(zNearFarV[0], zNearFarV[1], intrinsics, imgSizeV[0], imgSizeV[1]);
  GLuint list = createDisplayList(FM, fNum, VM, vNum);
  drawPatchToDepthBuffer(list, depthBuffer, imgSizeV[0], imgSizeV[1], zNearFarV);
  if (list) {
    glDeleteLists(list, 1);
    list = 0;
  }
}

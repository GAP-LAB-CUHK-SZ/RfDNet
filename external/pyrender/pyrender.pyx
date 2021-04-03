cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array() 


cdef extern from "offscreen.h":
  void renderDepthMesh(int *FM, int fNum, double *VM, int vNum, double *intrinsics, int *imgSizeV, double *zNearFarV, float *depthBuffer);


def render(double[:,::1] vertices, int[:,::1] faces, double[::1] cam_intr, int[::1] img_size):
  if vertices.shape[1] != 3:
    raise Exception('vertices must be a Mx3 double array')
  if faces.shape[1] != 3:
    raise Exception('faces must be a Mx3 int array')
  if cam_intr.shape[0] != 4:
    raise Exception('cam_intr must be a 4x1 double vector')
  if img_size.shape[0] != 2:
    raise Exception('img_size must be a 2x1 int vector')

  cdef double* VM = &(vertices[0,0])
  cdef int vNum = vertices.shape[0]
  cdef int* FM = &(faces[0,0])
  cdef int fNum = faces.shape[0]
  cdef double* intrinsics = &(cam_intr[0])
  cdef int* imgSize = &(img_size[0])

  cdef double znf[2]
  znf[0] = 1e10
  znf[1] = -1e10
  cdef double z
  for i in range(vNum):
    z = VM[2*vNum+i]
    if (z<znf[0]):
      znf[0] = z
    if (z>znf[1]):
      znf[1] = z

  znf[0] -= 0.1;
  znf[1] += 0.1;
  znf[0] = max(znf[0],0.1);
  znf[1] = max(znf[1],znf[0]+0.1);

  depth = np.empty((img_size[0], img_size[1]), dtype=np.float32)
  cdef float[:,::1] depth_view = depth
  cdef float* depthBuffer = &(depth_view[0,0])
  
  renderDepthMesh(FM, fNum, VM, vNum, intrinsics, imgSize, znf, depthBuffer);
  
  return depth

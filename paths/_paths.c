#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
 extern "C" {
#endif

static int dE(int L, int *x, int i, int j){
  /*
    Energy change if a spin (i,j) flips.
   */
  int s;

  s = x[((i-1+L)%L)*L + j] + x[((i+1)%L)*L + j]+ x[L*i+(j-1+L)%L] + x[L*i+(j+1)%L];
    
  return 2 * x[L*i+j] * s;
}

static PyObject *ising_energy(PyObject *self, PyObject *args) { 

  PyArrayObject *arr;
  int i, j, E=0, L, *x;

  if (!(PyArg_ParseTuple(args, "iO!", &L, &PyArray_Type, &arr))) {
    PyErr_SetString(PyExc_TypeError, "contiguous int array required");
    return NULL;
  }

  x = (int*) arr->data;
  
  for (i = 0; i < L; i++) {
    for (j = 0; j < L; j++) { 
      E += x[i*L+j] * (x[i*L + (j+1)%L] + x[((i+1)%L)*L + j]);
    }
  }

  return Py_BuildValue("i", -E);
}

static PyObject *ising_sample(PyObject *self, PyObject *args) { 

  PyArrayObject *arr;
  int i, j, n, a, acc=0, L, N, *x;
  double beta, E;

  if (!(PyArg_ParseTuple(args, "diiO!", &beta, &N, &L, &PyArray_Type, &arr))) {
    PyErr_SetString(PyExc_TypeError, "contiguous int array required");
    return NULL;
  }

  x = (int*) arr->data;
  
  for (n = 0; n < N; n++) {
    
    i = rand() % L;
    j = rand() % L;

    E = beta * dE(L, x, i, j);
    a = 1;

    if (E > 0.) {
      a = (int) (drand48() < exp(-E));
    }
    
    if (a) {
      x[i*L + j] *= -1;
    }
    acc += a;
  }
  
  return Py_BuildValue("i", acc);
}

static PyMethodDef methods[] = {
  {"ising_energy", (PyCFunction) ising_energy, 1},
  {"ising_sample", (PyCFunction) ising_sample, 1},
  {NULL, NULL}
};


void init_paths(void) {

  import_array();
  Py_InitModule("_paths", methods);

}

#ifdef __cplusplus
}
#endif

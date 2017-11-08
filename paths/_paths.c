#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
 extern "C" {
#endif

static int delta(int x, int y) {
  // Kronecker delta
  return (int) (x==y);
}

static int dE_ising(int L, int *x, int i, int j){
  /*
    Energy change if a spin (i,j) flips.
   */
  int s;

  s = x[((i-1+L)%L)*L + j] + x[((i+1)%L)*L + j]+ x[L*i+(j-1+L)%L] + x[L*i+(j+1)%L];
    
  return 2 * x[L*i+j] * s;
}

static int dE_potts(int L, int *x, int i, int j, int q){

  int q_old, q_neigh, E=0;

  q_old = x[i*L + j];

  q_neigh = x[i*L + (j+1)%L];
  E += delta(q,q_neigh) - delta(q_old, q_neigh);

  q_neigh = x[i*L + (j-1+L)%L];
  E += delta(q,q_neigh) - delta(q_old, q_neigh);

  q_neigh = x[((i+1)%L)*L + j];
  E += delta(q,q_neigh) - delta(q_old, q_neigh);

  q_neigh = x[((i-1+L)%L)*L + j];
  E += delta(q,q_neigh) - delta(q_old, q_neigh);

  return -E;
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

static PyObject *potts_energy(PyObject *self, PyObject *args) { 

  PyArrayObject *arr;
  int i, j, E=0, L, *x;

  if (!(PyArg_ParseTuple(args, "iO!", &L, &PyArray_Type, &arr))) {
    PyErr_SetString(PyExc_TypeError, "contiguous int array required");
    return NULL;
  }

  x = (int*) arr->data;
  
  for (i = 0; i < L; i++) {
    for (j = 0; j < L; j++) { 
      E += delta(x[i*L+j], x[i*L + (j+1)%L]) + delta(x[i*L+j], x[((i+1)%L)*L + j]);
    }
  }

  return Py_BuildValue("i", -E);
}

static PyObject *rbm_energy(PyObject *self, PyObject *args) { 

  PyArrayObject *x_arr, *a_arr, *b_arr, *W_arr;
  int i, j, m, n, *x;
  double *a, *b, *W, E=0.;

  if (!(PyArg_ParseTuple(args, "O!O!O!O!",
			 &PyArray_Type, &x_arr,
			 &PyArray_Type, &a_arr,
			 &PyArray_Type, &b_arr,
			 &PyArray_Type, &W_arr))) {
    PyErr_SetString(PyExc_TypeError, "contiguous int array required");
    return NULL;
  }

  m = a_arr->dimensions[0];
  n = b_arr->dimensions[0];

  x = (int*) (x_arr->data);

  a = (double*) (a_arr->data);
  b = (double*) (b_arr->data);
  W = (double*) (W_arr->data);
  
  for (i=0; i < m; i++) {
    if (x[i]) {
      E += a[i];
      for (j=0; j < n; j++) {
	if (x[m+j]) {
	  E += W[i*n + j];
	}
      }
    }
  }
  
  for (j=0, i=m; j < n; j++, i++) {
    if (x[i]) {
      E += b[j];
    }
  }

  return Py_BuildValue("d", -E);
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

    E = beta * dE_ising(L, x, i, j);
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

static PyObject *potts_sample(PyObject *self, PyObject *args) { 

  PyArrayObject *arr;
  int i, j, q, n, a, acc=0, L, Q, N, *x;
  double beta, E;

  if (!(PyArg_ParseTuple(args, "diiiO!", &beta, &N, &L, &Q, &PyArray_Type, &arr))) {
    PyErr_SetString(PyExc_TypeError, "contiguous int array required");
    return NULL;
  }

  x = (int*) arr->data;
  
  for (n = 0; n < N; n++) {
    
    i = rand() % L;
    j = rand() % L;
    q = rand() % Q;

    E = beta * dE_potts(L, x, i, j, q);
    a = 1;

    if (E > 0.) {
      a = (int) (drand48() < exp(-E));
    }
    
    if (a) {
      x[i*L + j] = q;
    }
    acc += a;
  }
  
  return Py_BuildValue("i", acc);
}

static PyMethodDef methods[] = {
  {"ising_energy", (PyCFunction) ising_energy, 1},
  {"ising_sample", (PyCFunction) ising_sample, 1},
  {"potts_energy", (PyCFunction) potts_energy, 1},
  {"potts_sample", (PyCFunction) potts_sample, 1},
  {"rbm_energy", (PyCFunction) rbm_energy, 1},
  {NULL, NULL}
};


void init_paths(void) {

  import_array();
  Py_InitModule("_paths", methods);

}

#ifdef __cplusplus
}
#endif

"""
Cython extension to calculate SED weights for star particles.

Calculates weights on an age / metallicity grid given the mass.

Args:
  z - 1 dimensional array of metallicity values (ascending)
  a - 1 dimensional array of age values (ascending)
  particle - 2 dimensional array of particle properties (N, 3)
  first column metallicity, second column age, third column mass

Returns:
  w - 2d array, dimensions (z, a), of weights to apply to SED array
"""


from bisect import bisect
import numpy as np
cimport numpy as np

cimport cython

ctypedef np.float32_t dtype_t
ctypedef np.float64_t dtype_s

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
def calculate_weights(np.ndarray[dtype_s, ndim=1] z,
 		      np.ndarray[dtype_s, ndim=1] a,
 		      np.ndarray[dtype_t, ndim=2] particle):

    cdef int ihigh, ilow
    cdef dtype_s ifrac, metal, age, mass

    cdef np.ndarray[dtype_s, ndim=3] w = np.zeros((len(z),len(a), 1))

    # simple test for sorted z and a arrays
    if z[0] > z[1]:
        raise ValueError('Metallicity array not sorted ascendingly')

    if a[0] > a[1]:
        raise ValueError('Age array not sorted ascendingly')
        

    for p in particle:

        metal = p[0]
        age = p[1]
        mass = p[2]

        ilow = bisect(z,metal)
        if ilow == 0:  # test if outside array range
            ihigh = ilow # set upper index to lower
            ifrac = 0 # set fraction to unity
        elif ilow == len(z):
            ilow -= 1 # lower index
            ihigh = ilow # set upper index to lower
            ifrac = 0
        else:
            ihigh = ilow # else set upper limit to bisect lower
            ilow -= 1 # and set lower limit to index below
            ifrac = (metal-z[(ilow)])/(z[ihigh]-z[ilow])

        jlow = bisect(a,age)
        if jlow == 0:
            jhigh = jlow
            jfrac = 0
        elif jlow == len(a):
            jlow -= 1
            jhigh = jlow
            jfrac = 0
        else:
            jhigh = jlow
            jlow -= 1
            jfrac = (age-a[(jlow)])/(a[jhigh]-a[jlow])

        w[ilow,jlow] += mass * (1-ifrac) * (1-jfrac)
        if ilow != ihigh:  # ensure we're not adding weights more than once when outside range
            w[ihigh,jlow] += mass * ifrac * (1-jfrac)
        if jlow != jhigh:
            w[ilow,jhigh] += mass * (1-ifrac) * jfrac
        if (ilow != ihigh) & (jlow != jhigh):
            w[ihigh,jhigh] += mass * ifrac * jfrac

    return w

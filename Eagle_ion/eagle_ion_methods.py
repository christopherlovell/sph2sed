import numpy as np
from bisect import bisect


def update_weights_raw(w,z,a,age,metal,mass):
    """
    Update weight matrix for a given particle. Values outside array sizes return an error.    
    N.B. make sure both z and a lists are sorted ascendingly
    """

    ilow = bisect(z,metal) - 1
    ifrac = (metal-z[(ilow)])/(z[ilow+1]-z[ilow])

    jlow = bisect(a,age)-1
    jfrac = (age-a[(jlow)])/(a[jlow+1]-a[jlow])

    w[ilow,jlow] += mass * (1-ifrac) * (1-jfrac)
    w[ilow+1,jlow] += mass * ifrac * (1-jfrac)
    w[ilow,jlow+1] += mass * (1-ifrac) * jfrac
    w[ilow+1,jlow+1] += mass * ifrac * jfrac

    return(w)


def update_weights(w,z,a,metal,age,mass):
    """
    Update weight matrix for a given particle. Values outside array sizes assign weights to edges of array.    
    N.B. make sure both z and a lists are sorted ascendingly

    w: weights array, empty (zeroed) numpy array (z*a)
    z: metallicity array (1D)
    a: age array(1D)
    metal, age mass: particle arrays for metallicity, age and mass

    """

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
    if ilow != ihigh & jlow != jhigh:
        w[ihigh,jhigh] += mass * ifrac * jfrac

    return(w)


def calculate_spectrum(halo,star_idx,sed_grid,age,metals,mass,x,y):
    """
    for a given halo index (halo) loop through star indices (star_idx) and calculate grid weights.
    Apply weights to full SED
    """
    ## calculate weights for given subhalo
    w = np.zeros((len(x),len(y)))  # initialise empty weights array
    for i in star_idx[halo]:  # loop through halo particles
        # filter star particle attributes for given subhalo
        w = update_weights(w,x,y,age[i],metals[i],mass[i])

    return([np.nansum(w.transpose()*sed_grid[i]) for i in range(len(sed_grid))])  # apply weights to sed spectrum





import numpy as np
from bisect import bisect

c = 2.99792E8

def calculate_xi_ion(Lnu, frequency):
    """
    Calculate LyC photon production efficiency
    
    Args:
        Lnu: Lsol Hz^-1
        frequency: Hz

    Returns:
        xi_ion: units [erg^-1 Hz]
    """
    
    # filter nan sed values
    mask = ~np.isnan(Lnu)
    Lnu = Lnu[mask]
    frequency = frequency[mask]

    Lnu_0p15 = Lnu[np.abs((c * 1e6 / frequency) - 0.15).argmin()]

    integ = Lnu / (6.626e-34 * frequency * 1e7) # energy in ergs
    integ /= Lnu_0p15

    b = c / 912e-10
    limits = frequency>b

    return np.trapz(integ[limits][::-1],frequency[limits][::-1])


def calculate_sed(subhalos, raw_sed, Z, a):
    """
    Calculate composite sed for an array of star particles.
    
    Args:
        subhalos - list of subhalos, each containing a dictionary containing the following: '
        raw_sed - SED array (Z * a * wavelength)
        Z - metallicity array
        a - age array
    
    Returns:
        sed: with the same length as raw_sed, returned with the same units
    """
    
    sed = [[] for i in range(len(subhalos))]
    
    for i in range(len(subhalos)):
        
        # ids = subhalos[i]['idx']
        metals = subhalos[i]['stellar metallicity']
        imass = subhalos[i]['initial stellar mass']
        age = subhalos[i]['stellar age']
        mass = subhalos[i]['stellar mass']
        
        w = np.zeros(raw_sed.shape[:2])  # initialise empty weights array of same shape as raw_sed

        # update weights
        for j in range(len(metals)): # loop through particles
            w = update_weights(w, Z, a[::-1], metals[j], age[j], imass[j])
            
        
        sed_temp = np.matmul(w,raw_sed) # multiply sed by weights grid
        sed[i] = sed_temp.sum(axis=(0,1)) # combine single composite spectrum
        
    
    return sed


def update_weights(w, z, a, metal, age,mass):
    """
    Update weight matrix for a given particle. Values outside array sizes assign weights to edges of array.    
    N.B. ensure both z and a arrays are sorted ascendingly

    Args:
        w - weights array, size (z*a)
        z - 1D metallicity array (sorted ascendingly) [Zsol]
        a - 1d age array (sorted ascendingly) []
        metal - particle metallicity [Zsol]
        age - particle age []
        mass - particle mass [10^6 Msol]

    Returns:
        w - updated weights array (z*a)
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


#def update_weights_raw(w,z,a,age,metal,mass):
#    """
#    Update weight matrix for a given particle. Values outside array sizes return an error.    
#    N.B. make sure both z and a lists are sorted ascendingly
#    """
#
#    ilow = bisect(z,metal) - 1
#    ifrac = (metal-z[(ilow)])/(z[ilow+1]-z[ilow])
#
#    jlow = bisect(a,age)-1
#    jfrac = (age-a[(jlow)])/(a[jlow+1]-a[jlow])
#
#    w[ilow,jlow] += mass * (1-ifrac) * (1-jfrac)
#    w[ilow+1,jlow] += mass * ifrac * (1-jfrac)
#    w[ilow,jlow+1] += mass * (1-ifrac) * jfrac
#    w[ilow+1,jlow+1] += mass * ifrac * jfrac
#
#    return(w)


# def calculate_spectrum(halo,star_idx,sed_grid,age,metals,mass,x,y):
#     """
#     for a given halo index (halo) loop through star indices (star_idx) and calculate grid weights.
#     Apply weights to full SED
#     """
#     ## calculate weights for given subhalo
#     w = np.zeros((len(x),len(y)))  # initialise empty weights array
#     for i in star_idx[halo]:  # loop through halo particles
#         # filter star particle attributes for given subhalo
#         w = update_weights(w,x,y,age[i],metals[i],mass[i])
# 
#     return([np.nansum(w.transpose()*sed_grid[i]) for i in range(len(sed_grid))])  # apply weights to sed spectrum


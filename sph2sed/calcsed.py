import numpy as np
from bisect import bisect

from . import weights
# import weights

c = 2.99792E8


def calculate_sed(sed, Z, a, p_metal, p_age, p_imass):
    """
    Calculate composite sed for an array of star particles.

    Args:
        sed - SED array (Z * a * wavelength) in units of L M^-1 (e.g., erg s^-1 Hz^-1 Msol^-1)
        Z - SED metallicity array
        a - SED age array
        p_metal (array) in same units as Z
        p_age (array)  in same units as a
        p_imass (array) in units of M (e.g. Msol)

    Returns:
        sed: with the same length as raw_sed, returned with units L (e.g. erg s^-1 Hz^-1)
    """

    raw_sed = sed.copy()

    if a[0] > a[1]:
        print("Age array not sorted ascendingly. Sorting...")
        a = a[::-1]  # sort age array ascendingly
        raw_sed = raw_sed[:,::-1,:]  # sort sed array age ascending
        
        
    if Z[0] > Z[1]:
        print("Metallicity array not sorted ascendingly. Sorting...")
        Z = Z[::-1]  # sort age array ascendingly
        raw_sed = raw_sed[::-1,:,:]  # sort sed array age ascending

        
    w = weights.calculate_weights(Z, a, np.array([p_metal, p_age, p_imass]).T)

    raw_sed = sed * w # multiply sed by weights grid
    raw_sed = np.nansum(raw_sed, (0,1)) # combine single composite spectrum

    return raw_sed


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

    # normalisation luminosity
    Lnu_0p15 = Lnu[np.abs((c * 1e6 / frequency) - 0.15).argmin()]

    integ = Lnu / (6.626e-34 * frequency * 1e7) # energy in ergs
    integ /= Lnu_0p15  # normalise

    b = c / 912e-10
    limits = frequency>b

    return np.trapz(integ[limits][::-1],frequency[limits][::-1])


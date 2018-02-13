import numpy as np

import pickle as pcl

import fsps

from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

Zsol = 0.0127

def grid_spectra(Nage=40, NZ=10, nebular=True, dust=False):
    """
    Generate grid of spectra with FSPS    
    """

    if dust:
        sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, logzsol=0.0, add_neb_emission=nebular,
                                    dust_type=2, dust2=0.2, dust1=0.0) # dust_type=1, dust2=0.2, dust1=0.2)
    else:
        sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, logzsol=0.0, add_neb_emission=nebular)


    wl = np.array(sp.get_spectrum(tage=13, peraa=True)).T[:,0]

    ages = np.logspace(-3.5, np.log10(cosmo.age(0).value-0.4), num=Nage, base=10)
    # ages = np.linspace(0, cosmo.age(0).value, Nage)

    scale_factors = cosmo.scale_factor([z_at_value(cosmo.lookback_time, age * u.Gyr) for age in ages])
    metallicities = np.linspace(3e-3, 5e-2, num=NZ) / Zsol

    spec = np.zeros((len(metallicities), len(ages), len(wl)))

    for i, Z in enumerate(metallicities):

        for j, a in enumerate(ages):

            sp.params['logzsol'] = np.log10(Z)
            if nebular: sp.params['gas_logz'] = np.log10(Z)

            spec[i,j] = sp.get_spectrum(tage=a, peraa=True)[1]   # Lsol / AA


    return spec, metallicities, scale_factors, wl



if __name__ == "__main__":

    spec, Z, age, wl = grid_spectra(nebular=False, dust=False)

    pickle = {'Spectra': spec, 'Metallicity': Z, 'Age': age, 'Wavelength': wl}

    pcl.dump(pickle, open('output/fsps.p','wb'))




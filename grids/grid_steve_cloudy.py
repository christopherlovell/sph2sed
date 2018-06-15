import numpy as np
import os
import pickle as pcl

from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

# Zsol = 0.0127
package_dir = os.path.dirname(os.path.abspath(__file__))

def pickle_grid(name='BC03-Padova1994_Salpeter_defaultCloudy', indir='steve_cloudy', outdir='intrinsic/output', lam_correct=1):
    
    dat = pcl.load(open("%s/%s.p"%(indir,name), 'rb'), encoding='latin1')

    if 'total' in dat.keys():
        Lnu = dat['total']    # Lnu (erg s^-1 Hz^-1)
    else:
        Lnu = dat['SED']
    
    
    lam = dat['lam'] * lam_correct  # microns
    nu = 3e8/(lam*1e-6)  # Hz
    L_AA = ((Lnu * nu) / (lam*1e4)) / 3.846E33  # Lsol / AA

    if 'ages' in dat.keys():
        ages = (dat['ages'] / 1e3) * u.Gyr  # Gyr
    elif 'log10age' in dat.keys():
        ## extra factor of 10 due to error in original pickle...
        ages = (pow(10, pow(10, dat['log10age'])) / 1e9)  * u.Gyr 
    else: raise ValueError('ages key not in dictionary')

    if 'metallicities' in dat.keys():
        Z = dat['metallicities']  # metal fraction
    elif 'Z' in dat.keys():
        Z = dat['Z']              # metal fraction
    else: raise ValueError('metallicites key not in dictionary')

    ## ensure SED shape is consistent
    if (L_AA.shape[0] == len(Z)) & (L_AA.shape[1] == len(ages)):
        pass
    elif (L_AA.shape[1] == len(Z)) & (L_AA.shape[0] == len(ages)):
        L_AA = np.swapaxes(L_AA, 0, 1)
    else: raise ValueError('SED array shape incompatible with age and metallicity array shapes')

    ## remove ages above age of universe
    age_mask = (ages < cosmo.age(0)) & (ages != 0.)
    ages = ages[age_mask]
    L_AA = L_AA[:,age_mask,:]

    ## convert to scale factor
    scale_factors = cosmo.scale_factor([z_at_value(cosmo.lookback_time, age) for age in ages])

    pickle = {'Spectra': L_AA, 'Metallicity': Z, 'Age': scale_factors, 'Wavelength': lam * 1e4}

    pcl.dump(pickle, open('%s/%s.p'%(outdir,name), 'wb'))


if __name__ == "__main__":

    pickle_grid(name='BC03-Padova1994_Salpeter_defaultCloudy')
    pickle_grid(name='BC03-Padova1994_Salpeter_defaultCloudy')
    pickle_grid(name='P2_2p351p3_defaultCloudy')
    pickle_grid(name='BPASSv2-bin_imf135all100_defaultCloudy')
    pickle_grid(name='P2_2p71p3_defaultCloudy')
    pickle_grid(name='BPASSv2_imf135all100_defaultCloudy')
    pickle_grid(name='P2_Chabrier_defaultCloudy')
    pickle_grid(name='FSPS_Salpeter_defaultCloudy')
    pickle_grid(name='P2_K01_defaultCloudy')
    pickle_grid(name='M05-rhb_Salpeter_defaultCloudy')
    pickle_grid(name='P2_Salpeter_defaultCloudy')
    pickle_grid(name='P2_2p01p3_defaultCloudy')

    # pickle_grid(name='BPASSv2_imf135all_100-bin')
    # pickle_grid(name='P2_2p0_1p3_ng')
    # pickle_grid(name='BC03_Padova1994_Salpeter_lr')
    # pickle_grid(name='M05_Salpeter_rhb')
    # pickle_grid(name='FSPS_default_Salpeter')
    # pickle_grid(name='P2_BG03_ng')
    # pickle_grid(name='P2_Chabrier_ng')
    # pickle_grid(name='P2_Salpeter_ng')
    # pickle_grid(name='P2_K01_ng')
    # pickle_grid(name='BPASSv2_imf135all_100')
    # pickle_grid(name='P2_2p35_1p3_ng')
    # pickle_grid(name='P2_2p7_1p3_ng') 
    
    # pickle_grid(name='BPASSv2.1.binary_ModSalpeter_300', lam_correct=1e-4) 



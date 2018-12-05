import numpy as np
import os
import pickle as pcl

from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

# Zsol = 0.0127

def pickle_grid(fname='BC03-Padova1994_Salpeter_defaultCloudy', indir='steve_cloudy', outdir='intrinsic/output', lam_correct=1, outname=None, cloudy=False):
    
    dat = pcl.load(open("%s/%s.p"%(indir,fname), 'rb'), encoding='latin1')

    if 'total' in dat.keys():
        Lnu = dat['total']    # Lnu (erg s^-1 Hz^-1)
    elif 'SED' in dat.keys():
        Lnu = dat['SED']
    elif ('nebular' in dat.keys()) & ('stellar' in dat.keys()):
        if cloudy:
            Lnu = dat['stellar'] + dat['nebular']
        else:
            Lnu = dat['stellar']
    elif 'L_nu' in dat.keys():
        Lnu = dat['L_nu']
    
    
    lam = dat['lam'] * lam_correct  # microns
    nu = 3e8/(lam*1e-6)  # Hz
    L_AA = ((Lnu * nu) / (lam*1e4)) / 3.846E33  # Lsol / AA

    if 'ages' in dat.keys():
        ages = (dat['ages'] / 1e3) * u.Gyr  # Gyr
    elif 'log10age' in dat.keys():
        if dat['log10age'].max() < 1.3:
            ## extra factor of 10 due to error in original pickle...
            ages = (pow(10, pow(10, dat['log10age'])) / 1e9)  * u.Gyr 
        else:
            ages = (pow(10, dat['log10age']) / 1e9)  * u.Gyr 
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

    if outname is None: outname =fname
    pcl.dump(pickle, open('%s/%s.p'%(outdir,outname), 'wb'))


if __name__ == "__main__":

    # pickle_grid(name='BC03-Padova1994_Salpeter_defaultCloudy')
    # pickle_grid(name='BC03-Padova1994_Salpeter_defaultCloudy')
    # pickle_grid(name='P2_2p351p3_defaultCloudy')
    # pickle_grid(name='BPASSv2-bin_imf135all100_defaultCloudy')
    # pickle_grid(name='P2_2p71p3_defaultCloudy')
    # pickle_grid(name='BPASSv2_imf135all100_defaultCloudy')
    # pickle_grid(name='P2_Chabrier_defaultCloudy')
    # pickle_grid(name='FSPS_Salpeter_defaultCloudy')
    # pickle_grid(name='P2_K01_defaultCloudy')
    # pickle_grid(name='M05-rhb_Salpeter_defaultCloudy')
    # pickle_grid(name='P2_Salpeter_defaultCloudy')
    # pickle_grid(name='P2_2p01p3_defaultCloudy')

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
    
    indir = '/research/astro/highz/SED/0.2/stellar/BuildGrid/SSP/grids/BPASSv2.2.1.binary'

    pickle_grid(fname='stellar', indir=indir+'/1p0_100', outname='BPASSv2.2.1.binary_1p0_100', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/1p0_300', outname='BPASSv2.2.1.binary_1p0_300', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/1p7_100', outname='BPASSv2.2.1.binary_1p7_100', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/1p7_300', outname='BPASSv2.2.1.binary_1p7_300', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/Chabrier_100', outname='BPASSv2.2.1.binary_Chabrier_100', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/Chabrier_300', outname='BPASSv2.2.1.binary_Chabrier_300', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/ModSalpeter_100', outname='BPASSv2.2.1.binary_ModSalpeter_100', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/ModSalpeter_300', outname='BPASSv2.2.1.binary_ModSalpeter_300', cloudy=False, lam_correct=1e-4)
    pickle_grid(fname='stellar', indir=indir+'/Salpeter_100', outname='BPASSv2.2.1.binary_Salpeter_100', cloudy=False, lam_correct=1e-4)
    
    indir = '/research/astro/highz/SED/0.5/nebular/BuildGrid/Z/grids/BPASSv2.2.1.binary'

    # pickle_grid(fname='nebular', indir=indir+'/ModSalpeter_300', outname='BPASSv2.2.1.binary_ModSalpeter_300_cloudy', cloudy=True, lam_correct=1e-4)

    pickle_grid(fname='nebular', indir=indir+'/1p0_100', outname='BPASSv2.2.1.binary_1p0_100_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/1p0_300', outname='BPASSv2.2.1.binary_1p0_300_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/1p7_100', outname='BPASSv2.2.1.binary_1p7_100_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/1p7_300', outname='BPASSv2.2.1.binary_1p7_300_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/Chabrier_100', outname='BPASSv2.2.1.binary_Chabrier_100_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/Chabrier_300', outname='BPASSv2.2.1.binary_Chabrier_300_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/ModSalpeter_100', outname='BPASSv2.2.1.binary_ModSalpeter_100_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/ModSalpeter_300', outname='BPASSv2.2.1.binary_ModSalpeter_300_cloudy', cloudy=True, lam_correct=1e-4)
    pickle_grid(fname='nebular', indir=indir+'/Salpeter_100', outname='BPASSv2.2.1.binary_Salpeter_100_cloudy', cloudy=True, lam_correct=1e-4)



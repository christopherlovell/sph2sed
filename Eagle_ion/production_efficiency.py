
from eagle_ion_methods import update_weights
from bisect import bisect

import matplotlib.pyplot as plt

import pickle as pcl
import numpy as np

import astropy.units as u
from astropy.cosmology import Planck13, z_at_value


# ----- Methods

def calculate_xi_ion(sed,frequency,mass):
    ## sed: erg cm^-2 s^-1 
    ## frequency: s^-1
    ## imass: initial stellar mass, 10^6 Msun

    mask = ~np.isnan(sed)
    sed = sed[mask]
    frequency = frequency[mask]

    Lnu = sed * np.pi * 4 * (3.086 * 10**19)**2 / frequency / mass

    Lnu_0p15 = Lnu[np.abs((c/frequency)*1E-4 - 0.15).argmin()]

    integ = Lnu / (6.626E-34 * frequency) / Lnu_0p15

    # integration limits
    b = c / 91.2E-9
    limits = frequency>b

    return np.trapz(integ[limits][::-1],frequency[limits][::-1])



# ----- physical constants
c = 2.99792E8

# ---- Load Subhalo properties

tag = '004_z008p075' #'003_z008p988' #'005_z007p050' # 

data = pcl.load(open('subfind_output/subhalo_ids_'+tag+'.p','rb'))#, encoding='latin1')

sim_z = data['header']['simulation redshift']
sim_age = data['header']['simulation age']
h = data['header']['hubble param']

subhalos = data['data']

del(data)


# ---- Load obscured sed
models = ['BC03_Padova1994_Salpeter_lr', 'BPASSv2_imf135all_100', 'BPASSv2_imf135all_100-bin', 'P2_Salpeter_ng', 'M05_Salpeter_rhb', 'FSPS_default_Salpeter']
#model =models[2]

for model in models:
  
  print(model)

  data = pcl.load(open('../Input_SPS/model_pickles/'+model+'.p', 'rb'), encoding='latin1')
  
  sed = data['SED']
  Z = data['metallicities']  # metallicity
  ages = data['ages']  # Myr
  wavelength = data['lam']  # [ AA ]
  frequency = c/(wavelength*1E-10)  # [ s^-1 ]
  
  del(data)
  
  #### calculate scale factor for all ages in SED tables ####
  
  particle_ages = sim_age - (ages * 10**-3)  # convert to age in simulation snapshot
  
  a = []
  for p in particle_ages:  # convert to scale factor (to match particle values)
      if(p > 0.0001):
          a.append(1 / (1 + (z_at_value(Planck13.age, p * u.Gyr))))
  
  
  #### Calculate flam for each subhalo ####
  
  Lnu = np.zeros((len(subhalos), len(wavelength)))
  xi_ion = np.zeros(len(subhalos))
  im = np.zeros(len(subhalos))
  m = np.zeros(len(subhalos))
  
  # integration limits
  b = c / 91.2E-9
  limits = frequency>b
  
  
  for i in range(len(subhalos)):
  
      #print(i)
  
      ids = subhalos[i]['idx']
      metals = subhalos[i]['stellar metallicity']
      imass = subhalos[i]['initial stellar mass']
      age = subhalos[i]['stellar age']
      mass = subhalos[i]['stellar mass']
  
      w = np.zeros((len(Z),len(a)))  # initialise empty weights array
  
      for j in range(len(ids)):
          w = update_weights(w, Z, a[::-1], metals[j], age[j], imass[j])
  
      raw_sed = sed[:,:len(a)]  # First filter age values that we actually have
      raw_sed = raw_sed * w  # multiply by weights grid
      raw_sed = np.column_stack(raw_sed.flatten())  # flatten grid, create column vector of sed arrays
      raw_sed = np.nansum(raw_sed,axis=1)  # sum sed values at each wavelength, ignoring nan values
  
      ## `raw_sed` has units [erg cm^-2 s^-1]. To convert to a luminosity, multiply 
      ##  by area of inside of sphere where radius of sphere = 10 pc = 3.086 * 10**19 cm
#      Lnu[i] = raw_sed * np.pi * 4 * (3.086 * 10**19)**2  # [erg s^-1]
#      Lnu[i] /= frequency                                 # [erg s^-1 Hz^-1]
#      Lnu[i] /= (sum(imass) * 10**6)              # [erg s^-1 Hz^-1 M_{\odot}^-1]
      Lnu[i] = xi_ion(raw_sed,frequency,sum(imass)*10**6)  
  
      Lnu_0p15 = Lnu[i,np.abs(wavelength*1E-4 - 0.15).argmin()]
  
      integ = Lnu[i] / (6.626E-34 * frequency) / Lnu_0p15
  
      xi_ion[i] = np.trapz(integ[limits],frequency[limits])
  
      im[i] = sum(imass)
      m[i] = sum(mass)
  
  
  
  output = {'galaxy sed': Lnu, 'xi_ion': xi_ion, 'initial stellar mass': im, 'stellar mass': m, 'wavelength': wavelength}
  
  pcl.dump(output, open('ion_output/ion_'+model+'_'+tag+'.p', 'wb'))
  
 


### xi_ion as a function of age


fig, axes = plt.subplots(3,2,sharex='col', sharey='row')

fig.text(0.5, 0.04, r"$log_{10}$(age)", ha='center')
fig.text(0.03, 0.5, r"$log_{10}(\xi_{ion})$", ha='center', rotation='vertical')

for m,ax in zip(models,axes.flatten()):

    data = pcl.load(open('../Input_SPS/model_pickles/'+m+'.p', 'rb'))#, encoding='latin1')

    sed = data['SED'] # erg cm^-2 s^-1
    Z = data['metallicities']  # metallicity
    ages = data['ages']  # Myr
    wavelength = data['lam']  # [ AA ]
    frequency = c/(wavelength*1E-6)  # [ s^-1 ]
        
    # sum sed values for a given metallicity / age
    
    xi_ion = np.zeros(sed.shape[1])
    
    for j in xrange(sed.shape[0]):
        for i in xrange(sed.shape[1]): 
            xi_ion[i] = np.log10(calculate_xi_ion(sed[j,i],frequency,1e9))
        
        ax.plot(np.log10(ages),xi_ion)

    ax.annotate(m,xy=(-1,min(xi_ion)))




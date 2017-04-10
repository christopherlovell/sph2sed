
##
## Plot Cloudy input and output spectra
##

import numpy as np
import matplotlib.pyplot as plt
import pickle

import math

#import os
#os.chdir("/home/chris/sussex/eagle_demo/SPS/")


model = 'BPASSv2_imf135all_100'
a = 1
z = 1

c = 2.99E8

# ------ atmosphere spectra


# flux units: \nu L_{\nu} / 4 \pi r_{0}^{2}
lam, m1, m2, m3, m4, m5, m6 = np.loadtxt('../Cloudy_output/'+model+'/'+str(a)+'_'+str(z)+'.cont', delimiter='\t', usecols = (0,1,2,3,4,5,6)).T

lam /= 1E10 # in m
freq = c / lam

# convert to L_{\nu}
def unit_conversion(m,frequency):
    m *= math.pi
    m /= frequency
    return(m)

m1 = unit_conversion(m1,freq)
m2 = unit_conversion(m2,freq)
m3 = unit_conversion(m3,freq)
m4 = unit_conversion(m4,freq)
m5 = unit_conversion(m5,freq)
m6 = unit_conversion(m6,freq)


#def unit_conversion(m,wavelength):
#    m *= math.pi
##    m *= 3.846E33
#    m *= lam

# ----- unobscured spectrum 

data = pickle.load(open('../Input_SPS/model_pickles/' + model + '.p','rb'))
#data = pickle.load(open('../Input_SPS/model_pickles/'+model+'.p','rb'),encoding='latin1')

sed = data['SED'] # L_{\nu} / L_{\odot}
Z = data['metallicities']
ages = data['ages']
wavelength = data['lam'] # \mu m

frequency = c / (wavelength*1e-6)


# convert to L_{\nu}
f = (sed[a,z]/3.827E33)

# ---- plots

norm = (10**7) / (1.99 * 10**30 * 10**6)  # / erg * M_{*}


plt.figure(1)

plt.xlim((0.001,10))
plt.ylim((-0.3,4))

plt.semilogx(wavelength,np.log10(f/norm),markersize=40,color='grey')

#plt.set_yscale('log')
#plt.set_xscale('log')

plt.semilogx(lam*1E6,np.log10((m1+1E-100)/norm))
plt.semilogx(lam*1E6,np.log10((m2+1E-100)/norm))
plt.semilogx(lam*1E6,np.log10((m3+1E-100)/norm))
plt.semilogx(lam*1E6,np.log10((m4+1E-100)/norm))

plt.axvline(x=0.0912)

plt.legend(['unobscured','incident','transmitted','nebular out','total'])
plt.xlabel(r'$\lambda / \mu m$')
plt.ylabel(r'$\nu L_{\nu}$')
#plt.show()

plt.savefig('spectrum.png', bbox_inches='tight', dpi=300)




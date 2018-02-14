
##
## Plot Cloudy output spectra
##

import numpy as np
import matplotlib.pyplot as plt
import pickle

import math

import pickle as pcl

model = 'BPASSv2_imf135all_100'
a = 1
z = 1

c = 2.99E8

# ------ atmosphere spectra
data = pcl.load(open('../Cloudy_output/output/'+model+'.p','rb'))#,encoding='latin1')


# flux units: \nu L_{\nu} / 4 \pi r_{0}^{2}
sed = data[0]
Z = data[1]
ages = data[2]
lam = data[3]

mask = ~np.isnan(sed[0,20])
f = sed[0,20][mask]
lam = lam[mask]


#f = sed[0,1]


lam /= 1E10 # in m
freq = c / lam


# convert to L_{\nu}
def unit_conversion(m,frequency,wavelength):
    m *= math.pi 
    m /= frequency #
#    m *= 3.846e33
#    m *= wavelength
    return(m)

f = unit_conversion(f,freq,lam)


# ---- plots
norm = 1#(10**7) / (1.99 * 10**30 * 10**6)  # / erg * M_{*}


plt.figure(1)

#plt.xlim((0.001,10))
#plt.ylim((-0.3,4))

#plt.set_yscale('log')
#plt.set_xscale('log')


plt.semilogx(lam[::-1]*1E6, np.log10((f+1E-100)/norm))
#plt.semilogx(lam[::-1]*1E6,np.log10(f+1E-100))

plt.axvline(x=0.0912)

#plt.legend(['unobscured','incident','transmitted','nebular out','total'])
plt.xlabel(r'$\lambda / \mu m$')
plt.ylabel(r'$\nu L_{\nu}$')
#plt.show()

plt.savefig('nebular_spectrum.png', bbox_inches='tight')




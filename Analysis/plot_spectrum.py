"""
Plot Cloudy input and output spectra

"""

import numpy as np
import pickle as pcl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['BC03_Padova1994_Salpeter_lr', 'BPASSv2_imf135all_100', 'BPASSv2_imf135all_100-bin', 
          'P2_Salpeter_ng', 'M05_Salpeter_rhb', 'FSPS_default_Salpeter']

model = models[0]
a = 1
z = 1

c = 2.99E8

cloudy = pcl.load(open('../Cloudy_output/output/'+model+'.p','rb'))  # ------ atmosphere spectra
intrinsic = pcl.load(open('../Input_SPS/pickles/' + model + '.p','rb'))  # ----- unobscured spectrum 

# ---- plot
plt.figure(1)

# plt.xlim((0.001,10))
# plt.ylim((-0.3,4))

plt.semilogx(intrinsic['Wavelength'], np.log10(intrinsic['SED'][a,z]), markersize=40, color='grey', label='intrinsic')
plt.semilogx(cloudy['Wavelength'], np.log10(cloudy['SED'][a,z]), label='Cloudy')

plt.axvline(x=912)

plt.legend()
# plt.legend(['unobscured','incident','transmitted','nebular out','total'])
plt.xlabel(r'$\lambda / \mu m$')
plt.ylabel(r'$\nu L_{\nu}$')
plt.savefig('spectrum.png', bbox_inches='tight', dpi=300)


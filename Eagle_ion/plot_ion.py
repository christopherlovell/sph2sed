
import pickle as pcl
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as sp

from matplotlib import cm
import brewer2mpl


# ----- physical constants
c = 2.99792E8

# ---- Load Subhalo properties

tag = '004_z008p075' #'005_z007p050' 

models = ['BC03_Padova1994_Salpeter_lr', 'BPASSv2_imf135all_100', 'BPASSv2_imf135all_100-bin', 'P2_Salpeter_ng', 'M05_Salpeter_rhb', 'FSPS_default_Salpeter']
models_pretty = ['BC03', 'BPASSv2 Single', 'BPASSv2 Binary', 'PEGASEv2', 'M05', 'FSPSv2.4']

model = models[2]
model_pretty = models_pretty[2]

data = pcl.load(open('ion_output/ion_'+model+'_'+tag+'.p','rb'))

Lnu = data['galaxy sed']
xi_ion = data['xi_ion']
imass = data['initial stellar mass'] * 1e6
mass = data['stellar mass'] * 1e6

del(data)


## ---- Calculate binned median

# Mass bins
massBinLimits = np.linspace(7.1, 11.1, 21)
massBins = np.logspace(7.2, 11.0, 20)


median, dump, dump = sp.binned_statistic(imass, xi_ion, statistic='median', bins=10**massBinLimits)


def percentile16(y,p=10):
    return(np.percentile(y,p))

def percentile84(y,p=84):
    return(np.percentile(y,p))

p16, dump, dump = sp.binned_statistic(imass, xi_ion, statistic=percentile16, bins=10**massBinLimits)

p84, dump, dump = sp.binned_statistic(imass, xi_ion, statistic=percentile84, bins=10**massBinLimits)


color = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors # define random colour palette


# plot xi_ion against stellar mass
fig, ax = plt.subplots()

# Remove top and right axes lines ("spines")
spines_to_remove = ['top', 'right']
for spine in spines_to_remove:
    ax.spines[spine].set_visible(False)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


plt.semilogx(imass, np.log10(xi_ion), '.', alpha=0.5, c=color[0])

mask = massBins < 5e9
plt.errorbar(massBins[mask], np.log10(median[mask]), yerr=[np.log10(median[mask])-np.log10(p16[mask]), np.log10(p84[mask])-np.log10(median[mask])], linestyle='none', marker = 'o', c=color[3], zorder=32)

plt.xlim(1.01e8, 1e10)

plt.xlabel(r"$M_{*} / M_{\odot}$")
plt.ylabel(r"$\mathrm{log}_{10}(\xi_{ion}/(\mathrm{erg}^{-1} \mathrm{Hz}))$")

plt.title('EAGLE LyC photon production efficiency, BPASSv2-bin$')

#plt.show()

plt.savefig('../../output/eagle_xi_ion.png', dpi=300)



## ---- SPS models 6-subplot

fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(3, 2, figsize=(7,9), sharex=True, sharey=True)

fig.subplots_adjust(wspace=0, hspace=0)

fig.text(0.2, 0.93, 'EAGLE LyC photon production efficiency, z=8')

fig.text(0.5, 0.04, r"$M_{*} / M_{\odot}$", ha='center')
fig.text(0.03, 0.5, r"$\mathrm{log}_{10}(\xi_{ion}/(\mathrm{erg}^{-1} \mathrm{Hz}))$", va='center', rotation='vertical')

axes = [ax1,ax2,ax3,ax4,ax5,ax6]

for model, model_pretty, ax in zip(models, models_pretty, axes):
    data = pcl.load(open('ion_output/ion_'+model+'_'+tag+'.p','rb'))

    Lnu = data['galaxy sed']
    xi_ion = data['xi_ion']
    imass = data['initial stellar mass'] * 1e6
    mass = data['stellar mass'] * 1e6

    median, dump, dump = sp.binned_statistic(imass, xi_ion, statistic='median', bins=10**massBinLimits)

    def percentile16(y,p=10):
        return(np.percentile(y,p))

    def percentile84(y,p=84):
        return(np.percentile(y,p))

    p16, dump, dump = sp.binned_statistic(imass, xi_ion, statistic=percentile16, bins=10**massBinLimits)
    p84, dump, dump = sp.binned_statistic(imass, xi_ion, statistic=percentile84, bins=10**massBinLimits)
    
    ax.set_xscale('log')
 
    ax.plot(imass, np.log10(xi_ion), '.', alpha=0.5, c=color[0])

    mask = massBins < 5e9
    ax.errorbar(massBins[mask], np.log10(median[mask]), yerr=[np.log10(median[mask])-np.log10(p16[mask]), np.log10(p84[mask])-np.log10(median[mask])], linestyle='none', marker = 'o', c=color[3], zorder=32)
    
    ax.set_xlim(1.01e8, 9.8e9)
    ax.set_ylim(25.01,26.3)

    ax.annotate(model_pretty, xy=(1,0.05), xycoords='axes fraction', horizontalalignment='right')
    
 
plt.savefig('../../output/eagle_xi-ion_all-models.png', dpi=300)


    
    


 
## ---- 2D histogram

xbins = 10**np.linspace(8, 10, 20)
ybins = np.linspace(25, 26, 15)

counts, xedges, yedges = np.histogram2d(mass, np.log10(xi_ion), bins=([xbins, ybins]))

extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

fig, ax = plt.subplots()
#ax.pcolormesh(xbins, ybins, counts.T)

levels = (1e1, 3e1, 1e2, 2e2)
cset = contour(counts.T, levels, origin='lower', colors=['black', 'green', 'blue', 'red'], linewidths=(1.9, 1.6, 1.5, 1.4), extent=extent)

#plt.clabel(cset, inline=1, fontsize=10, fmt='%1.0i')

for c in cset.collections:
    c.set_linestyle('solid')

    ax.set_xscale('log')
 
    plt.show()






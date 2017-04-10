import eagle as E

import numpy as np
import pandas as pd

import pickle

from bisect import bisect

import astropy.units as u
from astropy.cosmology import Planck13, z_at_value # TODO: confirm this is the cosmology used in EAGLE

import matplotlib.pyplot as plt



## ----- physical constants
c = 2.99792E8
h = 0.6777



def update_weights_raw(w,z,a,age,metal,mass):
    """
    Update weight matrix for a given particle. Values outside array sizes return an error.    
    N.B. make sure both z and a lists are sorted ascendingly
    """
    
    ilow = bisect(z,metal) - 1
    ifrac = (metal-z[(ilow)])/(z[ilow+1]-z[ilow])
     
    jlow = bisect(a,age)-1
    jfrac = (age-a[(jlow)])/(a[jlow+1]-a[jlow])
    
    w[ilow,jlow] += mass * (1-ifrac) * (1-jfrac)
    w[ilow+1,jlow] += mass * ifrac * (1-jfrac)
    w[ilow,jlow+1] += mass * (1-ifrac) * jfrac
    w[ilow+1,jlow+1] += mass * ifrac * jfrac
    
    return(w)
    


def update_weights(w,z,a,metal,age,mass):
    """
    Update weight matrix for a given particle. Values outside array sizes assign weights to edges of array.    
    N.B. make sure both z and a lists are sorted ascendingly
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
    



def calculate_spectrum(halo,star_idx,sed_grid,age,metals,mass,x,y):
    """
    for a given halo index (halo) loop through star indices (star_idx) and calculate grid weights.
    Apply weights to full SED
    """
    ## calculate weights for given subhalo
    w = np.zeros((len(x),len(y)))  # initialise empty weights array
    for i in star_idx[halo]:  # loop through halo particles
        # filter star particle attributes for given subhalo
        w = update_weights(w,x,y,age[i],metals[i],mass[i]) 
    
    return([np.nansum(w.transpose()*sed_grid[i]) for i in range(len(sed_grid))])  # apply weights to sed spectrum




##
## set simulation details
## - some lines commented out as the data they load is only required for calculating subhalo_star_ids 

tags = ['004_z008p075'] # test data tag
folder = '/gpfs/data/clovell/EagleData/L0100N1504'  # output folder
directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data' # test data location


# ----- read attributes
sim_z = E.readAttribute("SUBFIND", directory, tags[0], "/Header/Redshift")
sim_age = Planck13.age(sim_z).value # in Gyr


# ----- read id arrays
id_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/ParticleIDs")  # all star particle ids
#id_particle_sf = E.readArray("SUBFIND_PARTICLES", directory, tags[0], "/IDs/ParticleID")  # ids of subhalo particles

# ----- read group info arrays
#group_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/GroupNumber")  # particle groups
#group_sh = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/GroupNumber")  # subhalo groups

# ----- read particle properties
imass_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/InitialMass")
a_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/StellarFormationTime")
metal_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/SmoothedMetallicity")
mass_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/Mass")

# ----- convert masses to 10^6 M_sol
imass_star_ss *= 10**4 
#mass_star_ss *= (10**4)

# ----- read length arrays
#length_sf = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/SubLength")  # no of particles in subhalo
#offset_sf = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/SubOffset")  # offset where particles included starts
#lengthType_sf = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/SubLengthType")  # length of types of particles; unordered, therefore useless





#### Find particles IDs for each subhalo ####

# ---- only use halos with more than 100 star particles
#big_halos = np.where(lengthType_sf[:,4] > 10)[0]
#
#idx = [None] * len(big_halos)
#
#i = j = 0
#
## --- initialise
#group_no = group_sh[i]  
#mask = (group_ss == group_sh[i])
#halo_particles = id_star_ss[mask]
#
#for i in big_halos:
#
#    print(i)        
#
#    if(group_no != group_sh[i]):
#        mask = (group_ss == group_sh[i])  # mask for star particles in given subhalos FOF halo
#        halo_particles = id_star_ss[mask]  # filter star particles given mask
#      
#    subhalo_particles = id_particle_sf[offset_sf[i]:offset_sf[i]+length_sf[i]]  # return all particles in subhalo
#    filt = pd.Index(subhalo_particles).get_indexer(halo_particles) >= 0  # find star particles in subhalo from subsetted star particles in overall halo
#    
#    idx[j] = np.where(mask)[0][filt]
#    j += 1
#
#
#pickle.dump(idx, open('/gpfs/data/dc-love2/subhalo_star_ids.p','wb'))
#

# ---- load particle IDs for each subhalo
subhalo_data = '/gpfs/data/dc-love2/SPS_data/subhalo_starids_004_z008p075.p'
subhalo_star_ids = pickle.load(open(subhalo_data,'rb'))


# ---- load obscured sed
model = ['BC03_Padova1994_Salpeter_lr', 'BPASSv2_imf135all_100', 'BPASSv2_imf135all_100-bin', 'P2_Salpeter_ng', 'M05_Salpeter_rhb', 'FSPS_default_Salpeter']

data = pickle.load(open('../Cloudy_output/output/'+model[1]+'.p','rb')) #, encoding='latin1')

sed = data[0]
Z = data[1]  # metallicity
ages = data[2]  # Myr
wavelength = data[3]  # [ AA ]
frequency = c/(wavelength*1E-10)  # [ s^-1 ]


#### calculate scale factor for all ages in SED tables ####

particle_ages = sim_age - (ages * 10**-3)  # convert to age in simulation snapshot

a = []
for p in particle_ages:  # convert to scale factor (to match particle values)
    if(p > 0.0001):
        a.append(1 / (1 + (z_at_value(Planck13.age, p * u.Gyr))))




#### Single subhalo tests ####

mask = subhalo_star_ids[400]

# filter for all stellar properties
star_particle_th = id_star_ss[mask]
metal_star_th = metal_star_ss[mask]
imass_star_th = imass_star_ss[mask]
a_star_th = a_star_ss[mask]
mass_star_th = mass_star_ss[mask]

del(mask)


# ---- Calculate weights array 
w = np.zeros((len(Z),len(a)))  # initialise empty weights array

for i in range(len(star_particle_th)):
    w = update_weights(w,Z,a[::-1],metal_star_th[i],a_star_th[i],imass_star_th[i]) 


# `sed` is a 2d grid where each grid point is its own array.
raw_sed = sed[:,:len(a)]  # First filter age values that we actually have
raw_sed = raw_sed * w  # multiply by weights grid
raw_sed = np.column_stack(raw_sed.flatten())  # flatten grid, create column vector of sed arrays
raw_sed = np.nansum(raw_sed,axis=1)  # sum sed values at each wavelength, ignoring nan values


#pickle.dump(raw_sed,open('rawsed.p','wb'))
#raw_sed = pickle.load(open('rawsed.p','rb'))


## `raw_sed` has units [erg cm^-2 s^-1]
##  to convert to a luminosity, multiply by area of inside of sphere
##  where radius of sphere = 10 pc = 3.086 * 10**19 cm
Lnu = raw_sed * np.pi * 4 * (3.086 * 10**19)**2  # [erg s^-1]
Lnu /= frequency                                 # [erg s^-1 Hz^-1]
Lnu /= (sum(imass_star_th) * 10**6)              # [erg s^-1 Hz^-1 M_{\odot}^-1]



plt.figure(1)
#plt.ylim((19,24))
#plt.xlim((0.05,1.2))
plt.axvline(91.2/1E3)
plt.semilogx(wavelength/1E4, np.log10(Lnu))
#plt.xlabel(r'Wavelength ($\lambda / \nu m$)')
#plt.ylabel(r'$L_{\lambda}$')
plt.show()


plt.figure(1)
plt.semilogx(wavelength/1E4,np.log10(sed[0,0]))
plt.axvline(91.2/1E3)
plt.show()




# ---- Production efficiency
integ = Lnu / ((6.626E-34) * frequency)

# --- integration limits
b = c / 91.2E-9


mask = frequency > b

A = np.abs(wavelength-1500).argmin()

np.trapz(integ[mask], x=frequency[mask]) / Lnu[A]

del(mask)




#### Calculate flam for each subhalo ####

Lnu = np.zeros((len(subhalo_star_ids),len(wavelength)))
xi_ion = np.zeros(len(subhalo_star_ids))
m = np.zeros(len(subhalo_star_ids))

# integration limits
b = c / 91.2E-9
limits = frequency>b


for i in range(len(subhalo_star_ids)):
    
    print(i)
    
    mask = subhalo_star_ids[i]
    
    metals = metal_star_ss[mask]
    imass = imass_star_ss[mask]
    age = a_star_ss[mask]
    mass = mass_star_ss[mask]
    
    w = np.zeros((len(Z),len(a)))  # initialise empty weights array

    for j in range(len(mask)): 
        w = update_weights(w,Z,a[::-1],metals[j],age[j],imass[j]) 
    
    raw_sed = sed[:,:len(a)]  # First filter age values that we actually have
    raw_sed = raw_sed * w  # multiply by weights grid
    raw_sed = np.column_stack(raw_sed.flatten())  # flatten grid, create column vector of sed arrays
    raw_sed = np.nansum(raw_sed,axis=1)  # sum sed values at each wavelength, ignoring nan values

    ## `raw_sed` has units [erg cm^-2 s^-1]
    ##  to convert to a luminosity, multiply by area of inside of sphere
    ##  where radius of sphere = 10 pc = 3.086 * 10**19 cm
    Lnu[i] = raw_sed * np.pi * 4 * (3.086 * 10**19)**2  # [erg s^-1]
    Lnu[i] /= frequency                                 # [erg s^-1 Hz^-1]
    Lnu[i] /= (sum(imass_star_th) * 10**6)              # [erg s^-1 Hz^-1 M_{\odot}^-1]



    Lnu_0p15 = Lnu[i,np.abs(wavelength*1E-4 - 0.15).argmin()]

    integ = Lnu[i] / (6.626E-34 * frequency) / Lnu_0p15

    xi_ion[i] = np.trapz(integ[limits],frequency[limits])

    m[i] = sum(imass)



# plot xi_ion against stellar mass
plt.figure(1)
plt.semilogx(m*10**6, np.log10(xi_ion), '.')
plt.xlabel(r"$M_{*} / M_{\odot}$")
plt.ylabel(r"$\mathrm{log}_{10}(\xi_{ion}/(\mathrm{erg}^{-1} \mathrm{Hz}))$")
plt.show()


# 2D histogram

xbins = 10**np.linspace(8, 10, 20)
ybins = np.linspace(25, 26, 15)

counts, xedges, yedges = np.histogram2d(m*10**6, np.log10(xi_ion), bins=([xbins, ybins]))

extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

fig, ax = plt.subplots()
ax.pcolormesh(xbins, ybins, counts.T)

levels = (1e0,1e1,1.0e2, 1.0e3)
cset = contour(counts.T, levels, origin='lower', colors=['black', 'green', 'blue', 'red'], linewidths=(1.9, 1.6, 1.5, 1.4), extent=extent)
plt.clabel(cset, inline=1, fontsize=10, fmt='%1.0i')
for c in cset.collections:
    c.set_linestyle('solid')


ax.set_xscale('log')

plt.show()




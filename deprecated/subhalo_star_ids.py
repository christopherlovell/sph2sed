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
id_particle_sf = E.readArray("SUBFIND_PARTICLES", directory, tags[0], "/IDs/ParticleID")  # ids of subhalo particles

# ----- read group info arrays
group_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/GroupNumber")  # particle groups
group_sh = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/GroupNumber")  # subhalo groups

# ----- read particle properties
im_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/InitialMass")
a_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/StellarFormationTime")
metal_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/SmoothedMetallicity")
mass_star_ss = E.readArray("SNAPSHOT", directory, tags[0], "/PartType4/Mass")

# ----- convert masses to 10^6 M_sol
#im_star_ss *= 10**4
mass_star_ss *= (10**4) / h

# ----- read length arrays
length_sf = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/SubLength")  # no of particles in subhalo
offset_sf = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/SubOffset")  # offset where particles included starts
lengthType_sf = E.readArray("SUBFIND", directory, tags[0], "/Subhalo/SubLengthType")  # length of types of particles; unordered, therefore useless





#### Find particles IDs for each subhalo ####

# ---- only use halos with more than 100 star particles
big_halos = np.where(lengthType_sf[:,4] > 60)[0]

idx = [None] * len(big_halos)

i = j = 0

# --- initialise
group_no = group_sh[i]  
mask = (group_ss == group_sh[i])
halo_particles = id_star_ss[mask]

for i in big_halos:

    print(i)        

    if(group_no != group_sh[i]):
        mask = (group_ss == group_sh[i])  # mask for star particles in given subhalos FOF halo
        halo_particles = id_star_ss[mask]  # filter star particles given mask
      
    subhalo_particles = id_particle_sf[offset_sf[i]:offset_sf[i]+length_sf[i]]  # return all particles in subhalo
    filt = pd.Index(subhalo_particles).get_indexer(halo_particles) >= 0  # find star particles in subhalo from subsetted star particles in overall halo
    
    idx[j] = np.where(mask)[0][filt]
    j += 1


pickle.dump(idx, open('/gpfs/data/dc-love2/SPS_data/subhalo_starids_004_z008p075.p','wb'))


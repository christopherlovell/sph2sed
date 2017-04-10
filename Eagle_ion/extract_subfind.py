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


tag = '004_z008p075' # '003_z008p988' # '005_z007p050' 
directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data' # test data location


# ----- read attributes
sim_z = E.readAttribute("SUBFIND", directory, tag, "/Header/Redshift")
sim_age = Planck13.age(sim_z).value # in Gyr

h = E.readAttribute("SUBFIND", directory, tag, "/Header/HubbleParam")

# ----- read id arrays
id_star_ss = E.readArray("SNAPSHOT", directory, tag, "/PartType4/ParticleIDs")  # all star particle ids
id_particle_sf = E.readArray("SUBFIND_PARTICLES", directory, tag, "/IDs/ParticleID")  # ids of subhalo particles

# ----- read group info arrays
group_ss = E.readArray("SNAPSHOT", directory, tag, "/PartType4/GroupNumber")  # particle groups
group_sh = E.readArray("SUBFIND", directory, tag, "/Subhalo/GroupNumber")  # subhalo groups

group_mass = E.readArray("SUBFIND", directory, tag, "/Subhalo/Stars/Mass")

# ----- read particle properties
imass_star_ss = E.readArray("SNAPSHOT", directory, tag, "/PartType4/InitialMass")
a_star_ss = E.readArray("SNAPSHOT", directory, tag, "/PartType4/StellarFormationTime")
metal_star_ss = E.readArray("SNAPSHOT", directory, tag, "/PartType4/SmoothedMetallicity")
mass_star_ss = E.readArray("SNAPSHOT", directory, tag, "/PartType4/Mass")

# ----- convert masses to 10^6 M_sol
imass_star_ss *= 10**4
mass_star_ss *= (10**4)

# ----- read length arrays
length_sf = E.readArray("SUBFIND", directory, tag, "/Subhalo/SubLength")  # no of particles in subhalo
offset_sf = E.readArray("SUBFIND", directory, tag, "/Subhalo/SubOffset")  # offset where particles included starts
lengthType_sf = E.readArray("SUBFIND", directory, tag, "/Subhalo/SubLengthType")  # length of types of particles; unordered, therefore useless



#### Find particles IDs for each subhalo ####

# ---- only use halos with more than 100(0.01) [!subject to change] star particles
big_halos = np.where(group_mass > 0.009)[0]

# initialise index array
data = [None] * len(big_halos)


## Can't do a brute force search over star particle IDs as the arrays are too big.
## First filter by particles in FOF halo, then match on subsetted array.

for j,i in enumerate(big_halos):
    #print(i)        

    particle_in_fof = (group_ss == group_sh[i]) # mask for *all* particles in subgroups host FOF halo
    
    halo_star_particles = id_star_ss[particle_in_fof]  # filter for *star* particles in a given subhalos host FOF halo
      
    subhalo_particles = id_particle_sf[offset_sf[i]:offset_sf[i]+length_sf[i]]  # subset *all* particles in subhalo

    star_particles_subhalo = pd.Index(subhalo_particles).get_indexer(halo_star_particles) >= 0  # find *star* particles in subhalo from subsetted star particles in overall halo
    idx = np.where(particle_in_fof)[0][star_particles_subhalo] 

    data[j] = {'idx': idx}
    data[j]['initial stellar mass'] = imass_star_ss[ idx ]
    data[j]['stellar age'] = a_star_ss[ idx ]
    data[j]['stellar metallicity'] = metal_star_ss[ idx ]
    data[j]['stellar mass'] = mass_star_ss[ idx ]


output = {'header': {'simulation redshift': sim_z, 'simulation age': sim_age, 'hubble param': h}, 'data': data}


pickle.dump(output, open('subfind_output/subhalo_ids_'+tag+'.p','wb'))



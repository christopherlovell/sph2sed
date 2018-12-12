import numpy as np 
import pickle as pcl 
from sph2sed.model import sed 
from sklearn.decomposition import PCA 
from weights import calculate_weights                                          


sp = sed() 
sp.filename = '../../data/EAGLE_spectra/data/particle_data_eagle_z000p101.p' 
sp.load() 
 
spec = 'bc03_chab' 
zed = '%07.3f'%sp.redshift 
 
### Generate Spectra ### 
sp.grid_directory = '/cosma7/data/dp004/dc-love2/codes/sph2sed/grids/' 
sp.load_grid(spec) 
sp.redshift_grid(sp.redshift) 
 
for idx in list(sp.galaxies.keys()): 
    sp.galaxies[idx]['redshift'] = sp.redshift 

z = sp.redshift                                                                

grid = sp.grid['grid'][:,sp.grid['age_mask'][z],:]                             
# sp.grid['metallicity'] = sp.grid['metallicity'].astype(np.float32)
# sp.grid['age'][z] = sp.grid['age'][z].astype(np.float32)

# import pstats, cProfile
# import pyximport
# pyximport.install()
# 
# cProfile.runctx("calculate_weights(sp.grid['metallicity'], sp.grid['age'][z], np.array([sp.galaxies[0]['StarParticles']['Metallicity'], sp.galaxies[0]['StarParticles']['Age'], sp.galaxies[0]['StarParticles']['InitialMass']]).T)", globals(), locals(), "Profile.prof")
# 
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()

import line_profiler                                                           
profile = line_profiler.LineProfiler(calculate_weights)   

idx = 10

profile.runcall(calculate_weights, 
                sp.grid['metallicity'], 
                sp.grid['age'][z], 
                np.array([sp.galaxies[idx]['StarParticles']['Metallicity'], 
                          sp.galaxies[idx]['StarParticles']['Age'], 
                          sp.galaxies[idx]['StarParticles']['InitialMass']], dtype=np.float64).T) 

# profile.runcall(calculate_weights, 
#                 sp.grid['metallicity'].astype(np.float32), 
#                 sp.grid['age'][z].astype(np.float32), 
#                 np.array([sp.galaxies[0]['StarParticles']['Metallicity'], 
#                           sp.galaxies[0]['StarParticles']['Age'], 
#                           sp.galaxies[0]['StarParticles']['InitialMass']], dtype=np.float32).T) 

profile.print_stats()


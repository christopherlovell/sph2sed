import numpy as np
import pickle as pcl
import os
import sys
import random

from . import weights

import astropy.units as u
from astropy.cosmology import WMAP9, z_at_value

class sed:
    """
    Class encapsulating data structures and methods for generating spectral energy distributions (SEDs) from cosmological hydrodynamic simulations.
    """

    def __init__(self, details=''):

        self.package_directory = os.path.dirname(os.path.abspath(__file__))      # location of package
        self.grid_directory = os.path.split(self.package_directory)[0]+'/grids'  # location of SPS grids
        self.galaxies = {}     # galaxies info dictionary
        self.cosmo = WMAP9     # astropy cosmology
        self.age_lim = 0.1     # Young star age limit, used for resampling recent SF, Gyr
        self.details = details

        # check lookup tables exist, create if not
        print(self.package_directory)
        if os.path.isfile('%s/temp/lookup_table.txt'%self.package_directory):
            lookup_table = np.loadtxt('%s/temp/lookup_table.txt'%self.package_directory, dtype=np.float32)
            self.a_lookup = lookup_table[0]
            self.age_lookup = lookup_table[1]
        else:
            if query_yes_no("Lookup table not initialised. Would you like to do this now? (takes a minute or two)"):

                self.age_lookup = np.linspace(1e-6, self.age_lim, 5000)
                self.a_lookup = np.array([self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, a * u.Gyr)) for a in self.age_lookup], dtype=np.float32)

                np.savetxt('%s/temp/lookup_table.txt'%self.package_directory, np.array([self.a_lookup, self.age_lookup]))



    def refresh_directories(self):
        self.package_directory = os.path.dirname(os.path.abspath(__file__))      # location of package
        self.grid_directory = os.path.split(self.package_directory)[0]+'/grids'  # location of SPS grids



    def insert_galaxy(self, idx, p_initial_mass, p_age, p_metallicity, **kwargs):
        """
        Insert a galaxy into the `galaxy` dictionary.

        Args:
        idx - unique galaxy idx
        p_initial_mass - numpy array(N), particle initial mass (solar masses)
        p_age - numpy array(N), particle age (scale factor)
        p_metallicity - numpy array(N), particle metallicity (Z solar)
        """
      
        self.galaxies[idx] = {}

        # set some flags
        self.galaxies[idx]['resampled'] = False

        # add star particles
        self.galaxies[idx] = {'StarParticles': {'Age': None, 'Metallicity': None, 'InitialMass': None}}

        self.galaxies[idx]['StarParticles']['InitialMass'] = p_initial_mass
        self.galaxies[idx]['StarParticles']['Age'] = p_age
        self.galaxies[idx]['StarParticles']['Metallicity'] = p_metallicity

        # add some extra header info
        if kwargs is not None:
            self.insert_header(idx, **kwargs)



    def insert_header(self, idx, **kwargs):
        """
        Insert header information for a given galaxy

        Args:
        idx - unique galaxy identifier
        """

        if kwargs is not None:
            for key, value in kwargs.items():
                self.galaxies[idx][key] = value



    def load_grid(self, name='fsps'):
        """
        Load intrinsic spectra grid.

        Args:
        name - str, SPS model name to load
        """
         
        file_dir = '%s/intrinsic/output/%s.p'%(self.grid_directory,name)

        print("Loading %s model from: \n\n%s\n"%(name, file_dir))
        temp = pcl.load(open(file_dir, 'rb'))

        self.grid = temp['Spectra']
        self.metallicity = temp['Metallicity']
        
        self.age = temp['Age']  # scale factor
        self.lookback_time = self.cosmo.lookback_time((1. / self.age) - 1).value  # Gyr
        
        self.wavelength = temp['Wavelength']

        if self.age[0] > self.age[1]:
            print("Age array not sorted ascendingly. Sorting...\n")
            self.age = self.age[::-1]  # sort age array ascendingly
            self.lookback_time = self.lookback_time[::-1]
            self.grid = self.grid[:,::-1,:]  # sort sed array age ascending


        if self.metallicity[0] > self.metallicity[1]:
            print("Metallicity array not sorted ascendingly. Sorting...\n")
            self.metallicity = self.metallicity[::-1]  # sort Z array ascendingly
            self.grid = self.grid[::-1,:,:]  # sort sed array age ascending



    def resample_recent_sf(self, idx, sigma=5e-3, verbose=False):
        """
        Resample recently formed star particles.

        Star particles are much more massive than individual HII regions, leading to artificial Poisson scatter in the SED from recently formed particles.

        Args:
            idx (int) galaxy index
            age_lim (float) cutoff age in Gyr, lookback time
            sigma (float) width of resampling gaussian, Gyr
        """

        if self.resampled: raise ValueError('`resampled` flag already set; histories may already have been resampled.')

        # find age_cutoff in terms of the scale factor
        self.age_cutoff = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, self.age_lim * u.Gyr))
       
        mask = self.galaxies[idx]['StarParticles']['Age'] > self.age_cutoff

        if np.sum(mask) > 0:
            lookback_times = self.cosmo.lookback_time((1. / self.galaxies[idx]['StarParticles']['Age'][mask]) - 1).value
        else:
            if verbose: print("No young stellar particles! index: %s"%idx)
            return None

        resample_ages = np.array([], dtype=np.float32)
        resample_mass = np.array([], dtype=np.float32)
        resample_metal = np.array([], dtype=np.float32)
        
        for p_idx in np.arange(np.sum(mask)):
        
            N = int(self.galaxies[idx]['StarParticles']['InitialMass'][mask][p_idx] / 1e4)
            M_resample = np.float32(self.galaxies[idx]['StarParticles']['InitialMass'][mask][p_idx] / N)
        
            new_lookback_times = np.random.normal(loc=lookback_times[p_idx], scale=sigma, size=N)
        
            while True:
                
                # truncated Gaussian
                trunc_mask = (new_lookback_times < 0) | (new_lookback_times > 0.1)
        
                _lt = np.sum(trunc_mask)
        
                if not _lt:
                    break
        
                new_lookback_times[trunc_mask] = np.random.normal(loc=lookback_times[p_idx], scale=sigma, size=_lt)
        
        
            # lookup scale factor in tables 
            resample_ages = np.append(resample_ages, self.a_lookup[np.searchsorted(self.age_lookup, new_lookback_times)])
        
            resample_mass = np.append(resample_mass, np.repeat(M_resample, N))
        
            resample_metal = np.append(resample_metal, 
                                np.repeat(self.galaxies[idx]['StarParticles']['Metallicity'][mask][p_idx], N))
           


        self.galaxies[idx]['StarParticles']['Resampled'] = {}
        self.galaxies[idx]['StarParticles']['Resampled']['Age'] = resample_ages
        self.galaxies[idx]['StarParticles']['Resampled']['InitialMass'] = resample_mass
        self.galaxies[idx]['StarParticles']['Resampled']['Metallicity'] = resample_metal

        self.galaxies[idx]['StarParticles']['Resampled']['mask'] = mask

        # set 'resampled' flag
        self.galaxies[idx]['resampled'] = True



    def _calculate_weights(self, idx, resampled=False):


        if resampled & ('Resampled' in self.galaxies[idx]['StarParticles']):
            metal = self.galaxies[idx]['StarParticles']['Metallicity'][~self.galaxies[idx]['StarParticles']['Resampled']['mask']]
            metal = np.hstack([metal, self.galaxies[idx]['StarParticles']['Resampled']['Metallicity']])

            age = self.galaxies[idx]['StarParticles']['Age'][~self.galaxies[idx]['StarParticles']['Resampled']['mask']]
            age = np.hstack([age, self.galaxies[idx]['StarParticles']['Resampled']['Age']])
            
            imass = self.galaxies[idx]['StarParticles']['InitialMass'][~self.galaxies[idx]['StarParticles']['Resampled']['mask']]
            imass = np.hstack([imass, self.galaxies[idx]['StarParticles']['Resampled']['InitialMass']])
        else:
            metal = self.galaxies[idx]['StarParticles']['Metallicity']
            age = self.galaxies[idx]['StarParticles']['Age']
            imass = self.galaxies[idx]['StarParticles']['InitialMass'] 


        self._w = weights.calculate_weights(self.metallicity, self.age,
                                      np.array([metal,age,imass]).T)

        return self.grid * self._w
  


    def intrinsic_spectra(self, idx, key='Intrinsic Spectra', resampled=False):
        """
        Calculate composite intrinsic spectra.

        Args:
            idx (int) galaxy index
            ley (str) label to give generated spectra 
            resampled (bool) flg to use resampled young stellar particles (see `resample_recent_sf`)

        Returns:
            sed array, label `key`, with the same length as raw_sed, units L (e.g. erg s^-1 Hz^-1)
        """

        # self._w = weights.calculate_weights(self.metallicity, self.age, 
        #                               np.array([self.galaxies[idx]['StarParticles']['Metallicity'],
        #                                         self.galaxies[idx]['StarParticles']['Age'],
        #                                         self.galaxies[idx]['StarParticles']['InitialMass']]).T )

        # weighted_sed = self.grid * self._w                  # multiply sed by weights grid

        weighted_sed = self._calculate_weights(idx, resampled=resampled)

        self.galaxies[idx][key] = np.nansum(weighted_sed, (0,1))     # combine single composite spectrum



    def dust_screen(self, idx, resampled=False, tdisp=1e-2, tau_ism=0.33, tau_cloud=0.67, lambda_nu=5500, metal_dependent=False, verbose=False, name='Screen Spectra'):
        """
        Calculate composite spectrum with age dependent, and optional metallicity dependent, dust screen attenuation.

        Metallicity dependent dust screen requires inclusion of mass weighted star forming gas phase metallicity.

        Args:
            resampled (bool) flag, use resampled recently formed star particles (see `resample_recent_sf`)
            tdisp (float) birth cloud dispersion time, Gyr
            tau_ism (float) ISM optical depth at lambda_nu
            tau_cloud (float) birth cloud optical depth at lambda_nu
            lambda_nu (float) reference wavelength for optical depth values
            metal_dependent (bool) flag for applying metallicity dependent screen

        Returns:
            self

            Adds 'Screen Spectra' or 'Z-Screen Spectra' array to galaxy dict

        """

        # self._w = weights.calculate_weights(self.metallicity, self.age,
        #                               np.array([self.galaxies[idx]['StarParticles']['Metallicity'],
        #                                         self.galaxies[idx]['StarParticles']['Age'],
        #                                         self.galaxies[idx]['StarParticles']['InitialMass']]).T )

        # weighted_sed = self.grid * self._w
    
        weighted_sed = self._calculate_weights(idx, resampled=resampled)

        if metal_dependent:
        
            if verbose: print("Adding metallicity dependence to optical depth values")
           
            dependencies = ['sf_gas_metallicity','sf_gas_mass','stellar_mass']

            if not np.all([d in self.galaxies[idx] for d in dependencies]):
                raise ValueError('Required key missing from galaxy dict (idx %s)\ndependencies: %s'%(idx, dependencies))
            

            milkyway_mass = np.log10(6.43e10)
            Z_solar = 0.0134
            Z = 9.102 + np.log10(1 - np.exp((-1 * (milkyway_mass - 9.138)**0.513))) # Zahid+14 Mstar - Z relation (see Trayford+15)
            Z -= 8.69 # Convert from 12 + log()/H) -> Log10(Z / Z_solar) , Allende Prieto+01 (see Schaye+14, fig.13)
            Z = 10**Z

            ## Gas mass fractions
            gas_fraction = self.galaxies[idx]['sf_gas_mass'] / self.galaxies[idx]['stellar_mass']
            MW_gas_fraction = 0.1

            self.metallicity_factor = ((self.galaxies[idx]['sf_gas_metallicity'] / Z_solar) / Z) * (gas_fraction / MW_gas_fraction)
            tau_ism *= self.metallicity_factor
            tau_cloud *= self.metallicity_factor


        spec_A = np.nansum(weighted_sed[:,self.lookback_time < tdisp,:], (0,1))
        T = np.exp(-1 * (tau_ism + tau_cloud) * (self.wavelength / lambda_nu)**-0.7)
        spec_A *= T

        spec_B = np.nansum(weighted_sed[:,self.lookback_time >= tdisp,:], (0,1))
        T = np.exp(-1 * tau_ism * (self.wavelength / lambda_nu)**-0.7)
        spec_B *= T
    
        if metal_dependent:
            self.galaxies[idx][name] = spec_A + spec_B 
        else:
            self.galaxies[idx][name] = spec_A + spec_B 

        self.tau_ism = tau_ism
        self.tau_cloud = tau_cloud
        self.tdisp = tdisp
        self.lambda_nu = lambda_nu



    def recalculate_sfr(self, idx, time=0.1, label='sfr_100Myr'):
        """
        Recalculate SFR using particle data. 

        Adds an entry to the galaxies dict with key `label`.

        Args:
            idx (int) galaxy index
            time (float) lookback time over which to calculate SFR, Gyr
            label (str) label in galaxies dict to give SFR measure
        """

        # find age limit in terms of scale factor
        scalefactor_lim = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, time * u.Gyr))

        # mask particles below age limit
        mask = self.galaxies[idx]['StarParticles']['Age'] > scalefactor_lim

        # sum mass of particles (Msol), divide by time (yr)
        self.galaxies[idx][label] = np.sum(self.galaxies[idx]['StarParticles']['InitialMass'][mask]) / (scalefactor_lim * 1e9)  # Msol / yr



    def all_galaxies(self, method=None, **kwargs):
        """
        Apply a method to for all galaxies.

        Args:
            method (function) function to apply to all galaxies
        """

        if method is None:
            raise ValueError('method is None. Provide a valid method.') 
        else: 
            for key, value in self.galaxies.items():
                method(idx=key, **kwargs)



    def tidy(self, key=None):

        if key is None: raise ValueError('key is None, must specify key to remove from galaxies dictionary')
        elif isinstance(key, str):
            key = [key]
        
        for galid, value in self.galaxies.items():
             
            self.galaxies[galid] = {k: value[k] for k in value.keys() if k not in key}
            #      [k2 for k2 in self.galaxies[random.choice(list(self.galaxies.keys()))].keys() \
            #             if k2 not in key]}


    def load(self, encoding=None):
        if hasattr(self, 'filename'):
            f = open(self.filename, 'rb')
            if encoding is not None:
                tmp_dict = pcl.load(f, encoding=encoding)
            else:
                tmp_dict = pcl.load(f)
            f.close()          
            self.__dict__.update(tmp_dict)
        else:
            raise ValueError('Could not find "filename" in class instance.')
    


    def save(self):
        if hasattr(self, 'filename'):
            f = open(self.filename, 'wb')
            pcl.dump(self.__dict__, f)
            f.close()
        else:
            raise ValueError('Could not find "filename" in class instance.')


#     @staticmethod 
#     def calculate_xi_ion(Lnu, frequency):
#         """
#         Calculate LyC photon production efficiency
#     
#         Args:
#             Lnu: Lsol Hz^-1
#             frequency: Hz
#     
#         Returns:
#             xi_ion: units [erg^-1 Hz]
#         """
#     
#         # filter nan sed values
#         mask = ~np.isnan(Lnu)
#         Lnu = Lnu[mask]
#         frequency = frequency[mask]
#     
#         # normalisation luminosity
#         Lnu_0p15 = Lnu[np.abs((c * 1e6 / frequency) - 0.15).argmin()]
#     
#         integ = Lnu / (6.626e-34 * frequency * 1e7) # energy in ergs
#         integ /= Lnu_0p15  # normalise
#     
#         b = c / 912e-10
#         limits = frequency>b
#     
#         return np.trapz(integ[limits][::-1],frequency[limits][::-1])




def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


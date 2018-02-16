import numpy as np
import pickle as pcl
import os
import sys

from . import weights

import astropy.units as u
from astropy.cosmology import WMAP9, z_at_value

class sed:
    """
    Class encapsulating data and methods for generating spectral energy distributions (SEDs) from Smoothed Particle Hydrodynamics (SPH) simulations.
    """

    def __init__(self):

        self.package_directory = os.path.dirname(os.path.abspath(__file__))      # location of package
        self.grid_directory = os.path.split(self.package_directory)[0]+'/grids'  # location of SPS grids
        self.galaxies = {}     # galaxies info dictionary
        self.cosmo = WMAP9     # astropy cosmology
        self.age_lim = 0.1     # Young star age limit, used for resampling recent SF, Gyr

        # check lookup tables exist, create if not
        if os.path.isfile('%s/temp/lookup_tables.p'%self.package_directory):
            self.a_lookup, self.age_lookup = pcl.load(open('%s/temp/lookup_tables.p'%self.package_directory, 'rb'))
        else:
            if query_yes_no("Lookup tables not initialised. Would you like to do this now? (takes a minute or two)"):

                self.age_lookup = np.linspace(1e-6, self.age_lim, 5000)
                self.a_lookup = np.array([self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, a * u.Gyr)) for a in self.age_lookup], dtype=np.float32)

                pcl.dump([self.a_lookup, self.age_lookup], open('%s/temp/lookup_tables.p'%self.package_directory, 'wb'))


    def insert_galaxy(self, idx, imass, age, metallicity, **kwargs):
        """
        Insert a galaxy into the `galaxy` dictionary.

        Args:
        idx - unique galaxy idx
        imass - numpy array(N), particle initial mass (solar masses)
        age - numpy array(N), particle age (scale factor)
        metallicity - numpy array(N), particle metallicity (Z solar)

        """
        
        self.galaxies[idx] = {'Particles': {'Age': None, 'Metallicity': None, 'InitialMass': None}}

        self.galaxies[idx]['Particles']['InitialMass'] = imass
        self.galaxies[idx]['Particles']['Age'] = age
        self.galaxies[idx]['Particles']['Metallicity'] = metallicity

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


    def resample_recent_sf(self, idx, sigma=5e-3):
        """
        Resample recenly formed star particles.

        Star particles are much more massive than individual HII regions, leading to artificial Poisson scatter in the SED from recently formed particles.

        Args:
            idx (int) galaxy index
            age_lim (float) cutoff age in Gyr, lookback time
            sigma (float) width of resampling gaussian, Gyr
            
        Returns:
            
        """

        # find age_cutoff in terms of the scale factor
        self.age_cutoff = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, self.age_lim * u.Gyr))
       
        mask = self.galaxies[idx]['Particles']['Age'] > self.age_cutoff

        if np.sum(mask) > 0:
            lookback_times = self.cosmo.lookback_time((1. / self.galaxies[idx]['Particles']['Age'][mask]) - 1).value
        else:
            print("No young stellar particles!")
            return None

        resample_ages = np.array([], dtype=np.float32)
        resample_mass = np.array([], dtype=np.float32)
        resample_metal = np.array([], dtype=np.float32)
        
        for p_idx in np.arange(np.sum(mask)):
        
            N = int(self.galaxies[idx]['Particles']['InitialMass'][mask][p_idx] / 1e4)
            M_resample = np.float32(self.galaxies[idx]['Particles']['InitialMass'][mask][p_idx] / N)
        
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
                                np.repeat(self.galaxies[idx]['Particles']['Metallicity'][mask][p_idx], N))
           

        ## delete old particles 
        self.galaxies[idx]['Particles']['Age'] = np.delete(self.galaxies[idx]['Particles']['Age'], np.where(mask))
        self.galaxies[idx]['Particles']['InitialMass'] = np.delete(self.galaxies[idx]['Particles']['InitialMass'], np.where(mask))
        self.galaxies[idx]['Particles']['Metallicity'] = np.delete(self.galaxies[idx]['Particles']['Metallicity'], np.where(mask))
        
        ## resample new particles
        self.galaxies[idx]['Particles']['Age'] = np.concatenate([self.galaxies[idx]['Particles']['Age'], resample_ages])
        self.galaxies[idx]['Particles']['InitialMass'] = np.concatenate([self.galaxies[idx]['Particles']['InitialMass'], resample_mass])
        self.galaxies[idx]['Particles']['Metallicity'] = np.concatenate([self.galaxies[idx]['Particles']['Metallicity'], resample_metal])         





    def intrinsic_spectra(self, idx):
        """
        Calculate composite intrinsic spectra.

        Args:
            p_metal (array) in same units as Z
            p_age (array)  in same units as a
            p_imass (array) in units of M (e.g. Msol)

        Returns:
            sed: with the same length as raw_sed, returned with units L (e.g. erg s^-1 Hz^-1)
        """

        self._w = weights.calculate_weights(self.metallicity, self.age, 
                                      np.array([self.galaxies[idx]['Particles']['Metallicity'],
                                                self.galaxies[idx]['Particles']['Age'],
                                                self.galaxies[idx]['Particles']['InitialMass']]).T )

        weighted_sed = self.grid * self._w                  # multiply sed by weights grid
        self.galaxies[idx]['Intrinsic Spectra'] = np.nansum(weighted_sed, (0,1))     # combine single composite spectrum



    def dust_screen(self, idx, tdisp=1e-2, tau_ism=0.33, tau_cloud=0.67, lambda_nu=5500, metal_dependent=False, verbose=False):
        """
        Calculate composite spectrum with age dependent, and optional metallicity dependent, dust screen attenuation.

        Metallicity dependent dust screen requires inclusion of mass weighted star forming gas phase metallicity.

        Args:
            tdisp (float) birth cloud dispersion time, Gyr
            tau_ism (float) ISM optical depth at lambda_nu
            tau_cloud (float) birth cloud optical depth at lambda_nu
            lambda_nu (float) reference wavelength for optical depth values
            metal_dependent (bool) flag for applying metallicity dependent screen

        Returns:
            self

            Adds 'Screen Spectra' or 'Z-Screen Spectra' array to galaxy dict

        """

        self._w = weights.calculate_weights(self.metallicity, self.age,
                                      np.array([self.galaxies[idx]['Particles']['Metallicity'],
                                                self.galaxies[idx]['Particles']['Age'],
                                                self.galaxies[idx]['Particles']['InitialMass']]).T )

        weighted_sed = self.grid * self._w

        if metal_dependent:
        
            if verbose: print("Adding metallicity dependence to optical depth values")
            
            if 'metallicity' not in self.galaxies[idx]:
                raise ValueError('could not find key %s in galaxy dict'%'Metallicity')
            

            milkyway_mass = np.log10(6.43e10)
            Z_solar = 0.0134
            Z = 9.102 + np.log10(1 - np.exp((-1 * (milkyway_mass - 9.138)**0.513))) # Zahid+14 Mstar - Z relation (see Trayford+15)
            Z -= 8.69 # Convert from 12 + log()/H) -> Log10(Z / Z_solar) , Allende Prieto+01 (see Schaye+14, fig.13)
            Z = 10**Z

            self.metallicity_factor = (self.galaxies[idx]['metallicity'] / Z_solar) / Z
            tau_ism *= self.metallicity_factor
            tau_cloud *= self.metallicity_factor


        spec_A = np.nansum(weighted_sed[:,self.lookback_time < tdisp,:], (0,1))
        T = np.exp(-1 * (tau_ism + tau_cloud) * (self.wavelength / lambda_nu)**-0.7)
        spec_A *= T

        spec_B = np.nansum(weighted_sed[:,self.lookback_time >= tdisp,:], (0,1))
        T = np.exp(-1 * tau_ism * (self.wavelength / lambda_nu)**-0.7)
        spec_B *= T
    
        if metal_dependent:
            self.galaxies[idx]['Z-Screen Spectra'] = spec_A + spec_B 
        else:
            self.galaxies[idx]['Screen Spectra'] = spec_A + spec_B 

        self.tau_ism = tau_ism
        self.tau_cloud = tau_cloud
        self.tdisp = tdisp
        self.lambda_nu = lambda_nu



    def all_galaxies(self, method=None, **kwargs):
        """
        Calculate spectra for all galaxies.

        Args:
            method (function) spectra generating function to apply to all galaxies

        Returns:
            spectra for each galaxy in `self.galaxies`
        """

        if method is None:
            method = self.intrinsic_spectra
         
         
        for key, value in self.galaxies.items():
            # value['Spectra'] = method(idx=key, **kwargs)
            method(idx=key, **kwargs)



    def bin_histories(self, binning='linear', name='linear', Nbins=20):
            
        # initialise bins
        if binning == 'linear':
            binLimits = np.linspace(0, self.cosmo.age(0).value, Nbins+1)
            binWidth = binLimits[1] - binLimits[0]
            bins = np.linspace(binWidth/2, binLimits[-1] - binWidth/2, Nbins)
            binWidths = binWidth * 1e9        
        elif binning == 'log':
            upperBin = np.log10(self.cosmo.age(0).value * 1e9)
            binLimits = np.linspace(7.1, upperBin, Nbins+1)
            binWidth = binLimits[1] - binLimits[0]
            bins = np.linspace(7.1 + binWidth/2., upperBin - binWidth/2., Nbins)
            binWidths = 10**binLimits[1:] - 10**binLimits[:len(binLimits)-1]
            binLimits = 10**binLimits / 1e9
        else:
            raise ValueError('Invalid binning chosen, use either \'linear\' or \'log\'')
            
        
        # save binning info to header
        self.binning = {}
        self.binning[name] = {}
        self.binning[name]['binLimits'] = binLimits
        self.binning[name]['binWidth'] = binWidth
        self.binning[name]['bins'] = bins
        self.binning[name]['binWidths'] = binWidths

        ## convert binLimits to scale factor
        # need to first set age limits so that z calculation doesn't fail
        binLimits[binLimits < 1e-6] = 1e-6
        binLimits[binLimits > (self.cosmo.age(0).value - 1e-3)] = (self.cosmo.age(0).value - 1e-3)

        self.binning[name]['binLimits_sf'] = self.cosmo.scale_factor([z_at_value(self.cosmo.lookback_time, a) for a in binLimits * u.Gyr])


        # create a lookup table of ages to avoid calculating for every particle
        # age_lookup = np.linspace(1e-4, cosmo.age(0).value - 1e-2, 20000)
        # z_lookup = [z_at_value(cosmo.lookback_time, a) for a in age_lookup * u.Gyr]

        for key, value in self.galaxies.items():

        #     # particles = value['history']

        #     # calculate age in Gyr using lookup table
        #     # formation_age = age_lookup[np.searchsorted(z_lookup, scale_factor_to_z(particles['formationTime']))]

        #     counts, dummy = np.histogram(formation_age, 
        #             bins=binLimits, weights=particles['InitialStellarMass']);  # weights converts age to SFR

            # weights converts age to SFR
            # bins mjst increase monotonically, so reverse for now
            counts, dummy = np.histogram(value['Particles']['Age'], 
                bins=self.binning[name]['binLimits_sf'][::-1], weights=value['Particles']['InitialMass']);

            value[name] = {}

            # reverse counts to fit original
            value[name]['SFH'] = counts[::-1] / self.binning[name]['binWidths'] # divide by bin width in (Giga)years to give SFR in Msol / year

        # 
        # return pickle 



    def load(self):
        if hasattr(self, 'filename'):
            f = open(self.filename, 'rb')
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

import numpy as np
import pickle as pcl
import os

from . import weights

from astropy.cosmology import WMAP9

class sed:
    """
    Class encapsulating data and methods for generating spectral energy distributions (SEDs) from Smoothed Particle Hydrodynamics (SPH) simulations.
    """

    def __init__(self):
        self.package_directory = os.path.dirname(os.path.abspath(__file__))
        self.galaxies = {}
        self.cosmo = WMAP9



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

        file_dir = '%s/intrinsic/output/%s.p'%(self.package_directory,name)

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


    def dust_screen(self, idx, tdisp=1e-2, tau_ism=0.33, tau_cloud=0.67, lambda_nu=5500, metal_dependent=False):
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
        
            print("Adding metallicity dependence to optical depth values")
            
            if 'Metallicity' not in self.galaxies[idx]:
                raise ValueError('could not find key %c in galaxy dict'%'Metallicity')
            

            milkyway_mass = np.log10(6.43e10)
            Z_solar = 0.0134
            Z = 9.102 + np.log10(1 - np.exp((-1 * (milkyway_mass - 9.138)**0.513))) # Zahid+14 Mstar - Z relation (see Trayford+15)
            Z -= 8.69 # Convert from 12 + log()/H) -> Log10(Z / Z_solar) , Allende Prieto+01 (see Schaye+14, fig.13)
            Z = 10**Z

            self.metallicity_factor = (self.galaxies[idx]['Metallicity'] / Z_solar) / Z
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



    def all_galaxies_intrinsic_spectra(self, verbose=False):
        """
        Calculate spectra for all galaxies.
        """
        
        # for key, value in self.galaxies.data.items():
        #     if verbose: 
        #         print(key)
            
        for key, value in self.galaxies.data.items():
            
            value['Spectra'] = self.intrinsic_spectra(key)

        

#     def load(self):
#         f = open(self.filename, 'rb')
#         tmp_dict = cPickle.load(f)
#         f.close()          
#     
#         self.__dict__.update(tmp_dict) 
#     
#     
#     def save(self):
#         f = open(self.filename, 'wb')
#         cPickle.dump(self.__dict__, f, 2)
#         f.close()

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


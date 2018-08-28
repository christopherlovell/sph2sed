import numpy as np
import pickle as pcl
import os
import sys
import random

from . import weights

import astropy.units as u
from astropy.cosmology import WMAP9, z_at_value

import pyphot

c = 2.9979e18  # AA s^-1

class sed:
    """
    Class encapsulating data structures and methods for generating spectral energy distributions (SEDs) from cosmological hydrodynamic simulations.
    """

    def __init__(self, details=''):

        self.package_directory = os.path.dirname(os.path.abspath(__file__))          # location of package
        self.grid_directory = os.path.split(self.package_directory)[0]+'/grids'      # location of SPS grids
        self.filter_directory = os.path.split(self.package_directory)[0]+'/filters'  # location of filters
        self.galaxies = {}     # galaxies info dictionary
        self.spectra = {}      # spectra info dictionary
        self.cosmo = WMAP9     # astropy cosmology
        self.age_lim = 0.1     # Young star age limit, used for resampling recent SF, Gyr
        self.details = details


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

        # add placeholder for spectra
        self.galaxies[idx]['Spectra'] = {}

        # add star particles
        self.galaxies[idx]['StarParticles'] = {'Age': None, 'Metallicity': None, 'InitialMass': None}

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



    def initialise_grid(self, name):
        
        self.grid[name]['lookback_time'] = self.cosmo.lookback_time((1. / self.grid[name]['age']) - 1).value  # Gyr

        if self.grid[name]['age'][0] > self.grid[name]['age'][1]:
            print("Age array not sorted ascendingly. Sorting...\n")
            
            # sort age array ascendingly
            self.grid[name]['age'] = self.grid[name]['age'][::-1]  
            self.grid[name]['lookback_time'] = self.grid[name]['lookback_time'][::-1]
            
            # sort sed array age ascending
            self.grid[name]['grid'] = self.grid[name]['grid'][:,::-1,:] 

        if self.grid[name]['metallicity'][0] > self.grid[name]['metallicity'][1]:
            print("Metallicity array not sorted ascendingly. Sorting...\n")
            
            # sort Z array ascendingly
            self.grid[name]['metallicity'] = self.grid[name]['metallicity'][::-1] 
            
            # sort sed array age ascending
            self.grid[name]['grid'] = self.grid[name]['grid'][::-1,:,:] 



    def load_grid(self, name='fsps'):
        """
        Load intrinsic spectra grid.

        Args:
            name - str, SPS model name to load
        """
         
        file_dir = '%s/intrinsic/output/%s.p'%(self.grid_directory,name)

        print("Loading %s model from: \n\n%s\n"%(name, file_dir))
        temp = pcl.load(open(file_dir, 'rb'))

        name = 0.0
        self.grid = {name: {'grid': None, 'age': None, 'metallicity':None}}

        self.grid[name]['grid'] = temp['Spectra']
        self.grid[name]['metallicity'] = temp['Metallicity']
        self.grid[name]['age'] = temp['Age']  # scale factor
        self.grid[name]['wavelength'] = temp['Wavelength']

        self.initialise_grid(name)



    def redshift_grid(self, z, z0=0.0):
        """
        Redshift grid ages, return new grid

        Args:
            z (float) redshift
        """

        if z == 0.:
            print("No need to initialise new grid, z = 0")
            return None
        else:
            
            self.grid[z] = self.grid[z0].copy()

            observed_lookback_time = self.cosmo.lookback_time(z).value

            # redshift of age grid values
            age_grid_z = [z_at_value(self.cosmo.scale_factor, a) for a in self.grid[z]['age']]
            # convert to lookback time
            age_grid_lookback = np.array([self.cosmo.lookback_time(z).value for z in age_grid_z])
            # add observed lookback time
            age_grid_lookback += observed_lookback_time

            # truncate age grid by age of universe
            age_mask = age_grid_lookback < self.cosmo.age(0).value
            age_grid_lookback = age_grid_lookback[age_mask]

            # convert new lookback times to redshift
            age_grid_z = [z_at_value(self.cosmo.lookback_time, t * u.Gyr) for t in age_grid_lookback]
            # convert redshift to scale factor
            age_grid = self.cosmo.scale_factor(age_grid_z)

            self.grid[z]['age'] = age_grid
            self.grid[z]['grid'] = self.grid[z]['grid'][:,age_mask,:]
            self.initialise_grid(z)



    def create_lookup_table(self, z, resolution=5000):
        
        # if query_yes_no("Lookup table not initialised. Would you like to do this now? (takes a minute or two)"):

        lookback_time = self.cosmo.lookback_time(z).value # Gyr

        self.age_lookup = np.linspace(lookback_time, lookback_time + self.age_lim, resolution)
        self.a_lookup = np.array([self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, a * u.Gyr)) for a in self.age_lookup], dtype=np.float32)

        filename = "%s/temp/lookup_%s_z%03.0fp%.3s_lim%1.0fp%.3s.txt"%(self.package_directory, self.cosmo.name, z, str(z%1)[2:], 
                                                                       self.age_lim, str("%.3f"%(self.age_lim%1))[2:])

        np.savetxt(filename, np.array([self.a_lookup, self.age_lookup]))


    def load_lookup_table(self, z):

        filename = "%s/temp/lookup_%s_z%03.0fp%.3s_lim%1.0fp%.3s.txt"%(self.package_directory, self.cosmo.name, z, str(z%1)[2:],
                                                                       self.age_lim, str("%.3f"%(self.age_lim%1))[2:])

        if os.path.isfile(filename):
            lookup_table = np.loadtxt(filename, dtype=np.float32)
            self.a_lookup = lookup_table[0]
            self.age_lookup = lookup_table[1]
        else:
            print("lookup table not initialised for this cosmology / redshift / age cutoff. initialising now (make take a couple of minutes)")
            self.create_lookup_table(z)


    def resample_recent_sf(self, idx, sigma=5e-3, verbose=False):
        """
        Resample recently formed star particles.

        Star particles are much more massive than individual HII regions, leading to artificial Poisson scatter in the SED from recently formed particles.

        Args:
            idx (int) galaxy index
            age_lim (float) cutoff age in Gyr, lookback time
            sigma (float) width of resampling gaussian, Gyr
        """

        # if self.resampled: raise ValueError('`resampled` flag already set; histories may already have been resampled. If not, reset flag.')
        
        if 'redshift' not in self.galaxies[idx]: raise ValueError('redshift not defined for this galaxy')

        if ('a_lookup' not in self.__dict__) | ('age_lookup' not in self.__dict__):
            self.load_lookup_table(self.galaxies[idx]['redshift'])

        if (self.a_lookup.min() > self.cosmo.scale_factor(self.galaxies[idx]['redshift'])) |\
                (self.a_lookup.max() < self.cosmo.scale_factor(self.galaxies[idx]['redshift'])):

            print('Lookup table out of range. Reloading')
            self.load_lookup_table(self.galaxies[idx]['redshift'])
            
        lookback_time_z0 = np.float32(self.cosmo.lookback_time(self.galaxies[idx]['redshift']).value)
        lookback_time_z1 = np.float32((self.cosmo.lookback_time(self.galaxies[idx]['redshift']) + self.age_lim * u.Gyr).value)

        # find age_cutoff in terms of the scale factor
        self.age_cutoff = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, lookback_time_z1 * u.Gyr))

        mask = self.galaxies[idx]['StarParticles']['Age'] > self.age_cutoff
        N = np.sum(mask)

        if N > 0:
            lookback_times = self.cosmo.lookback_time((1. / self.galaxies[idx]['StarParticles']['Age'][mask]) - 1).value
            if verbose: print("Young stellar particles: %s"%N)
        else:
            if verbose: print("No young stellar particles! index: %s"%idx)
            return None

        resample_ages = np.array([], dtype=np.float32)
        resample_mass = np.array([], dtype=np.float32)
        resample_metal = np.array([], dtype=np.float32)
        
        for p_idx in np.arange(N):
        
            n = int(self.galaxies[idx]['StarParticles']['InitialMass'][mask][p_idx] / 1e4)
            M_resample = np.float32(self.galaxies[idx]['StarParticles']['InitialMass'][mask][p_idx] / n)
        
            new_lookback_times = np.random.normal(loc=lookback_times[p_idx], scale=sigma, size=n)
        
            while True:
                
                # truncated Gaussian
                trunc_mask = (new_lookback_times < lookback_time_z0) | (new_lookback_times > lookback_time_z1)
        
                _lt = np.sum(trunc_mask)
        
                if not _lt:
                    break
        
                new_lookback_times[trunc_mask] = np.random.normal(loc=lookback_times[p_idx], scale=sigma, size=_lt)
        
       
            # lookup scale factor in tables 
            resample_ages = np.append(resample_ages, self.a_lookup[np.searchsorted(self.age_lookup, new_lookback_times)])
        
            resample_mass = np.append(resample_mass, np.repeat(M_resample, n))
        
            resample_metal = np.append(resample_metal, 
                                np.repeat(self.galaxies[idx]['StarParticles']['Metallicity'][mask][p_idx], n))
           


        self.galaxies[idx]['StarParticles']['Resampled'] = {}
        # make extra sure it's float32
        self.galaxies[idx]['StarParticles']['Resampled']['Age'] = resample_ages.astype(np.float32)
        self.galaxies[idx]['StarParticles']['Resampled']['InitialMass'] = resample_mass.astype(np.float32)
        self.galaxies[idx]['StarParticles']['Resampled']['Metallicity'] = resample_metal.astype(np.float32)

        self.galaxies[idx]['StarParticles']['Resampled']['mask'] = mask

        # set 'resampled' flag
        self.galaxies[idx]['resampled'] = True



    def _calculate_weights(self, idx, resampled=False):
        """
        Calculate weights matrix from stellar particles.

        Args:
            idx (int) galaxy index
            resampled (bool) whether to use resampled star particles
        """


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


        # Load grid appropriate for redshift
        if 'redshift' in self.galaxies[idx]:
            z = self.galaxies[idx]['redshift']

            if z not in self.grid.keys():

                print('Generating new grid')
                self.redshift_grid(z)
        else: 
            z = 0.0

        
        self._w = weights.calculate_weights(self.grid[z]['metallicity'], 
                                            self.grid[z]['age'],
                                            np.array([metal,age,imass]).T)

        return self.grid[z]['grid'] * self._w
  




    def intrinsic_spectra(self, idx, key='Intrinsic', resampled=False):
        """
        Calculate composite intrinsic spectra.

        Args:
            idx (int) galaxy index
            ley (str) label to give generated spectra 
            resampled (bool) flg to use resampled young stellar particles (see `resample_recent_sf`)

        Returns:
            sed array, label `key`, with the same length as raw_sed, units L (e.g. erg s^-1 Hz^-1)
        """

        weighted_sed = self._calculate_weights(idx, resampled=resampled)
        
        if key not in self.spectra:  # save spectra info
            self.spectra[key] = {'lambda': self.grid[0.0]['wavelength'], 'units': 'Lsol / AA', 'scaler': None}

        self.galaxies[idx]['Spectra'][key] = np.nansum(weighted_sed, (0,1))     # combine single composite spectrum


    def highz_dust(self, idx, gamma = -1.0, lambda_nu = 5500, key='highz', resampled=False):
        """
        Simple dust model for high-redshift

        Args:
            idx (int) galaxy index
            gamma (float) exponent
            lambda_nu (float) pivot wavelength
            key (str) label
            resampled (bool) whether to use resampled star particles or not
        """
        
        weighted_sed = self._calculate_weights(idx, resampled=resampled)

        if key not in self.spectra:  # save spectra info
            self.spectra[key] = {'lambda': self.grid[0.0]['wavelength'], 'units': 'Lsol / AA', 'scaler': None}

        wl = self.grid[self.galaxies[idx]['redshift']]['wavelength']

        dependencies = ['sf_gas_metallicity','sf_gas_mass','stellar_mass']

        if not np.all([d in self.galaxies[idx] for d in dependencies]):
            raise ValueError('Required key missing from galaxy dict (idx %s)\ndependencies: %s'%(idx, dependencies))
   
        tau_0 = 1e-8

        tau_V = tau_0 * self.galaxies[idx]['sf_gas_metallicity'] * self.galaxies[idx]['sf_gas_mass']

        T = np.exp(-tau_V * ((wl / lambda_nu)**gamma))
    
        self.galaxies[idx]['Spectra'][key] = np.nansum(weighted_sed, (0,1)) * T         




    def dust_screen(self, idx, resampled=False, tdisp=1e-2, tau_ism=0.33, tau_cloud=0.67, lambda_nu=5500, metal_dependent=False, verbose=False, key='Screen', custom_redshift=None):
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

        # if key not in self.spectra:  # save spectra info
        #     self.spectra[key] = {'lambda': self.wavelength, 'units': 'Lsol / AA', 'scaler': None}
    
        weighted_sed = self._calculate_weights(idx, resampled=resampled)

        wl = self.grid[self.galaxies[idx]['redshift']]['wavelength']
        lb = self.grid[self.galaxies[idx]['redshift']]['lookback_time']

        if custom_redshift is None:
            z = self.galaxies[idx]['redshift']
        else:
            z = custom_redshift
        
        
        if metal_dependent:
        
            if verbose: print("Adding metallicity dependence to optical depth values")
           
            dependencies = ['sf_gas_metallicity','sf_gas_mass','stellar_mass']

            if not np.all([d in self.galaxies[idx] for d in dependencies]):
                raise ValueError('Required key missing from galaxy dict (idx %s)\ndependencies: %s'%(idx, dependencies))
           
            milkyway_mass = 10.8082109              
            Z_solar = 0.0134
            M_0 = np.log10(1 + z) * 2.64 + 9.138
            Z_0 = 9.102
            beta = 0.513
            logOHp12 = Z_0 + np.log(1 - np.exp(-1 * (10**(milkyway_mass - M_0))**beta)) # Zahid+14 Mstar - Z relation (see Trayford+15)
            Z = 10**(logOHp12 - 8.69)  # Convert from 12 + log()/H) -> Log10(Z / Z_solar) , Allende Prieto+01 (see Schaye+14, fig.13)

            ## Gas mass fractions
            gas_fraction = self.galaxies[idx]['sf_gas_mass'] / self.galaxies[idx]['stellar_mass']
            MW_gas_fraction = 0.1

            metallicity_factor = ((self.galaxies[idx]['sf_gas_metallicity'] / Z_solar) / Z) * (gas_fraction / MW_gas_fraction)

            self.galaxies[idx]['metallicity_factor'] = metallicity_factor            

            tau_ism *= metallicity_factor
            tau_cloud *= metallicity_factor


        spec_A = np.nansum(weighted_sed[:,lb < tdisp,:], (0,1))
        T = np.exp(-1 * (tau_ism + tau_cloud) * (wl / lambda_nu)**-1.3)  # da Cunha+08 slope of -1.3
        spec_A *= T

        spec_B = np.nansum(weighted_sed[:,lb >= tdisp,:], (0,1))
        T = np.exp(-1 * tau_ism * (wl / lambda_nu)**-0.7)
        spec_B *= T
    
        if metal_dependent:
            self.galaxies[idx]['Spectra'][key] = spec_A + spec_B 
        else:
            self.galaxies[idx]['Spectra'][key] = spec_A + spec_B 

        self.tau_ism = tau_ism
        self.tau_cloud = tau_cloud
        self.tdisp = tdisp
        self.lambda_nu = lambda_nu


    def recalculate_sfr(self, idx, z, time=0.1, label='sfr_100Myr'):
        """
        Recalculate SFR using particle data. 

        Adds an entry to the galaxies dict with key `label`.

        Args:
            idx (int) galaxy index
            z (float) redshift of galaxy
            time (float) lookback time over which to calculate SFR, Gyr
            label (str) label in galaxies dict to give SFR measure
        """

        # find age limit in terms of scale factor
        scalefactor_lim = self.cosmo.scale_factor(z_at_value(self.cosmo.lookback_time, (self.cosmo.lookback_time(z).value + time) * u.Gyr))

        # mask particles below age limit
        mask = self.galaxies[idx]['StarParticles']['Age'] > scalefactor_lim

        # sum mass of particles (Msol), divide by time (yr)
        self.galaxies[idx][label] = np.sum(self.galaxies[idx]['StarParticles']['InitialMass'][mask]) / (time * 1e9)  # Msol / yr



    def _initialise_pyphot(self):
        self.filters = pyphot.get_library()


    @staticmethod
    def flux_frequency_units(L, wavelength):
        """
        Convert from Lsol AA^-1 -> erg s^-1 cm^-2 Hz^-1 in restframe (10 pc)
    
        Args:
            L [Lsol AA^-1]
            wavelength [AA]
        """
        c = 2.9979e18    # AA s^-1
        d_factor = 1.1964951828447575e+40  # 4 * pi * (10 pc -> cm)**2
        Llamb = L * 3.826e33               # erg s^-1 AA^1
        Llamb /= d_factor                  # erg s^-1 cm^-2 AA^-1
        return Llamb * (wavelength**2 / c)   # erg s^-1 cm^-2 Hz^-1 
   

    # @staticmethod
    # def luminosity_units(f, wavelength):
    #     """
    #     Convert from erg s^-1 cm^-2 Hz^-1 -> Lsol Hz^-1
    #     """
    # 
    #     d = (10 * u.pc).to(u.cm).value
    #     L = f * (4 * np.pi * d**2)           # erg s^-1 Hz^-1
    #     return L 


    @staticmethod
    def mean_luminosity(L, lamb, filt_trans, filt_lamb, trans_lim=1e-3):
        mask = filt_trans > trans_lim
        L_interp = np.interp(filt_lamb[mask], lamb, L)
        return np.mean(L_interp * filt_trans[mask])

    
    @staticmethod
    def photo(fnu, lamb, filt_trans, filt_lamb):
        """
        Absolute magnitude
    
        Args:
            fnu - flux [erg s^-1 cm^-2 Hz^-1]
            lamb - wavelength [AA]
            ftrans - filter transmission over flamb
            flamb - filter wavelength [AA]
        """

        nu = c / lamb
        
        filt_nu = c / filt_lamb

        if ~np.all(np.diff(filt_nu) > 0):
            filt_nu = filt_nu[::-1]
            filt_trans = filt_trans[::-1]
    
        ftrans_interp = np.interp(nu, filt_nu, filt_trans)
        a = np.trapz(fnu * ftrans_interp / nu, nu)
        b = np.trapz(ftrans_interp / nu, nu)

        mag = -2.5 * np.log10(a / b) - 48.6  # AB
    
        # AB magnitude, mean monochromatic flux
        return mag, a/b
    

    def calculate_photometry(self, idx, filter_name='SDSS_g', spectra='Intrinsic', wavelength=None, verbose=False, restframe_filter=True, redshift=None, user_filter=None):
        """
        Args:
            idx (int) galaxy index
            filter_name (string) name of filter in pyphot filter list
            spectra (string) spectra identifier, *assuming restframe luminosity per unit wavelength*
            wavelength (array) if None, use the self.wavelenght definition, otherwise define your own wavelength array
            verbose (bool)
        """

        if redshift is None:
            z = self.galaxies[idx]['redshift']
        else:
            z = redshift

        if 'filters' not in self.__dict__:
            if verbose: print('Loading filters..')
            self._initialise_pyphot()

        if 'Photometry' not in self.galaxies[idx]: 
            self.galaxies[idx]['Photometry'] = {}

        # get pyphot filter
        if user_filter is not None:
            f = user_filter
        else:
            f = self.filters[filter_name]

        if restframe_filter:
            filt_lambda = np.array(f.wavelength.to('Angstrom'))
        else:
            filt_lambda = np.array(f.wavelength.to('Angstrom')) / (1+z)


        if wavelength is None:
            wavelength = self.grid[self.galaxies[idx]['redshift']]['wavelength'] # AA
        

        spec = self.galaxies[idx]['Spectra'][spectra].copy()
        spec = self.flux_frequency_units(spec, wavelength)

        write_name = "%s %s"%(filter_name, spectra)
        M = self.photo(spec, wavelength, f.transmit, filt_lambda)[0]
        self.galaxies[idx]['Photometry'][write_name] = M


    @staticmethod
    def Madau96(lamz, z):
        """
        courtesy of Ciaran
        """
    
        lam = lamz/(1.+z)
    
        L = [1216.0,1026.0,973.0,950.0]
        A = [0.0036,0.0017,0.0012,0.00093]
    
        expteff = np.zeros(lamz.shape)
        teff = np.zeros(lamz.shape)
    
        for l,a in zip(L,A):
            teff[lam<l] += a * (lamz[lam<l]/l)**3.46
    
    
        expteff[lam>L[-1]] = np.exp(-teff[lam>L[-1]])
    
        s = [lam<=L[-1]]
    
        expteff[s] = np.exp(-(teff[s]+ 0.25*(lamz[s]/L[-1])**3*((1+z)**0.46-(lamz[s]/L[-1])**0.46)+9.4*(lamz[s]/L[-1])**1.5*((1+z)**0.18-(lamz[s]/L[-1])**0.18)-0.7*(lamz[s]/L[-1])**3*((lamz[s]/L[-1])**(-1.32)-(1+z)**(-1.32))+0.023*((lamz[s]/L[-1])**1.68-(1+z)**1.68)))
    
        expteff[lam>L[0]] = 1.0
    
        return expteff



    def igm_absoprtion(self, idx, key, observed_wl, z):
        """
        Apply IGM absoprtion according to the Madau+96 prescription

        Args:
            idx - galaxy index
            key - spectra key
            observer_wl - *restframe* wavelength
            z - redshift            
        """

        T = self.Madau96(observed_wl, z)

        self.galaxies[idx]['Spectra'][key] *= T     



    def all_galaxies(self, method=None, **kwargs):
        """
        Apply a method to all galaxies.

        Args:
            method (function) function to apply to all galaxies
            **kwargs (dict) (argument, value) pairs to pass to `method`
        """

        if method is None:
            raise ValueError('method is None. Provide a valid method.') 
        else: 
            for idx in self.galaxies.keys():
                method(idx=idx, **kwargs)



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


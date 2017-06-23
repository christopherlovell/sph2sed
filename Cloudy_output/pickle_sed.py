
import numpy as np
import pickle

out_dir = 'output/'
c = 2.99792458e8

models = ['BC03_Padova1994_Salpeter_lr','BPASSv2_imf135all_100','BPASSv2_imf135all_100-bin','FSPS_default_Salpeter','M05_Salpeter_rhb','P2_Salpeter_ng']


def cloudy_unit_conversion(m, wavelength):
    """
    Cloudy output units: nu Lnu / (4 pi r^2)

    Convert to: Lnu
    """
    m *= np.pi
    m /= (c/wavelength)
    return(m)


for model in models:

    model_pickle = '../Input_SPS/model_pickles/'+model+'.p'

    print 'Loading', model, ' SED pickle'
    data = pickle.load(open(model_pickle,'rb'))#, encoding='latin1')

    Z = data['metallicities']
    ages = data['ages']

    print 'Finding wavelength values'
    wavelengths = np.array([])

    for i,z in enumerate(Z):
        for j,a in enumerate(ages):
            
            lam = np.loadtxt(model+'/'+str(i)+'_'+str(j)+'.cont', delimiter='\t', usecols = [0]) 
            wavelengths = np.unique(np.concatenate([wavelengths,lam]))


    wavelengths = np.sort(wavelengths)[::-1]

    print 'Saving SED'

    cloudy_sed = np.zeros((len(Z),len(ages)), dtype=np.ndarray)

    for i,z in enumerate(Z):
        for j,a in enumerate(ages):
            
            lam, flambda = np.loadtxt(model+'/'+str(i)+'_'+str(j)+'.cont', delimiter='\t', usecols = [0,4]).T
        
            # create an empty sed array 
            sed = np.empty(len(wavelengths))
            sed[:] = np.nan

            # populate with available sed values
            sed[np.in1d(wavelengths,lam)] = flambda

            cloudy_sed[i,j] = cloudy_unit_conversion(sed, wavelengths*1e-10)


    print 'Pickle post-cloudy SED'
    data = {'SED': cloudy_sed,  # Lnu
            'Metallicity': Z,   
            'Age': ages,  # Myr
            'Wavelength': wavelengths}  # 1e10 m

    pickle.dump(data,open(out_dir+model+'.p','wb'), protocol=2)


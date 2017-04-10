
import numpy as np
import pickle


out_dir = 'output/'

models = ['BC03_Padova1994_Salpeter_lr','BPASSv2_imf135all_100','BPASSv2_imf135all_100-bin','FSPS_default_Salpeter','M05_Salpeter_rhb','P2_Salpeter_ng']

c = 3E8


for model in models:

    model_pickle = '../Input_SPS/model_pickles/'+model+'.p'

    # --- load pickled sed

    print('Loading '+model+' SED pickle')

    data = pickle.load(open(model_pickle,'rb'))#, encoding='latin1')

    # sed = data['SED']
    Z = data['metallicities']
    ages = data['ages']

    ### load post-cloudy obscured sed

    # find all wavelength values across all metallicities and ages
    print('Finding wavelength values')

    wavelengths = np.array([])

    for i,z in enumerate(Z):
        for j,a in enumerate(ages):
            
            lam = np.loadtxt(model+'/'+str(i)+'_'+str(j)+'.cont', delimiter='\t', usecols = [0])
            
            wavelengths = np.unique(np.concatenate([wavelengths,lam]))


    wavelengths = np.sort(wavelengths)[::-1]

    # save sed values for each metallicity and age
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

            cloudy_sed[i,j] = sed


    print('Pickle post-cloudy SED')

    #data = [cloudy_sed,Z,ages,wavelengths]
    data = {'SED': cloudy_sed,
            'Metallicity': Z,
            'Age': ages,
            'Wavelength': wavelengths}

    pickle.dump(data,open(out_dir+model+'.p','wb'), protocol=2)




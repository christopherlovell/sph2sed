"""
Convert pickles from model_pickles directory to same structure and units as cloudy pickle output


"""


import pickle as pcl

models = ['BC03_Padova1994_Salpeter_lr', 'BPASSv2_imf135all_100', 'BPASSv2_imf135all_100-bin', 'P2_Salpeter_ng', 'M05_Salpeter_rhb', 'FSPS_default_Salpeter']

for model in models:

  print(model)

  with open("model_pickles/%s.p"%model, 'rb') as f:
    data = pcl.load(f)#, encoding='latin1')
  
  out = {}
  out['Age'] = data['ages']   # Myr
  out['SED'] = data['SED']  # Lnu [W Hz^-1]
  out['SED'] /= 1e7  # Lnu [erg s^-1 Hz^-1]  

  out['Metallicity'] = data['metallicities']   # M_odot yr^-1
  out['Wavelength'] = data['lam'] * 1e4  # AA 
 
  with open("pickles/%s.p"%model, 'wb') as f:
    pcl.dump(out, f)


"""
Convert pickles from model_pickles directory to same structure as cloudy pickle output


"""


import pickle as pcl

models = ['BC03_Padova1994_Salpeter_lr', 'BPASSv2_imf135all_100', 'BPASSv2_imf135all_100-bin', 'P2_Salpeter_ng', 'M05_Salpeter_rhb', 'FSPS_default_Salpeter']

for model in models:

  print model

  data = pcl.load(open('model_pickles/'+model+'.p','rb'))
  
  out = {}
  out['Age'] = data['ages']
  out['SED'] = data['SED']
  out['Metallicity'] = data['metallicities']
  out['Wavelength'] = data['lam']
  
  pcl.dump(out, open('pickles/'+model+'.p', 'w'))


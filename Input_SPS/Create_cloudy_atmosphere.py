
## Given an SED (from an SPS model, for example), generate a cloudy atmosphere 
## grid. Can optionally generate an array of input files for selected parameters.

import numpy as np
import pickle
import os
import subprocess


## ------- Custom Parameters
cloudy_directory = '~/c13.03/' # '/home/chris/src/c13.03/'
wd = '/research/astro/highz/Students/Chris/sed_modelling/'  # working directory

models = ['BC03_Padova1994_Salpeter_lr','BPASSv2_imf135all_100','BPASSv2_imf135all_100-bin','FSPS_default_Salpeter','M05_Salpeter_rhb','P2_Salpeter_ng']

c = 2.9979E8

for model in models:

    model_pickle = 'model_pickles/'+model+'.p'
    out_dir = wd+'Cloudy_input/'+model+'/'
    cloudy_dir = wd+'Cloudy_output/'+model+'/'
    cloudy_output_dir = cloudy_dir + 'output/'

    # --- check directories exist

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if not os.path.isdir(cloudy_dir):
	os.makedirs(cloudy_dir)

    if not os.path.isdir(cloudy_output_dir):
	os.makedirs(cloudy_output_dir)


    # --- load pickled sed

    print('Loading '+model+' SED pickle')

    data = pickle.load(open(model_pickle,'rb'))#, encoding='latin1')
    
    sed = data['SED']
    Z = data['metallicities']
    ages = data['ages']
    wavelength = data['lam']

    frequency = c / (wavelength * 1E-6)

    ages *= 10**6  # Myr -> yr
    ages[ages == 0.0] = 1.


    # ---- TEMP check for negative values and amend
    # The BPASS binary sed has a couple of erroneous negative values, possibly due to interpolation errors
    # Here we set the Flux to the average of each neighbouring wavelength value

    mask = np.asarray(np.where(sed < 0.))
    for i in range(mask.shape[1]):
        sed[mask[0,i],mask[1,i],mask[2,i]] = sed[mask[0,i],mask[1,i],mask[2,i]-1]+sed[mask[0,i],mask[1,i],mask[2,i]+1]/2
        


    ##
    ## Write .ascii file
    ##

    print('Writing .ascii')
    
    outfile = model+'.ascii'

    output = []
    
    target = open(out_dir+model+'.ascii','w')    
    
    output.append("20060612\n")  # magic number
    output.append("2\n")  # ndim
    output.append("2\n")  # npar
    
    ## First parameter MUST be log otherwise Cloudy throws a tantrum
    output.append("par1\n")  # label par 1 
    output.append("par2\n")  # label par 2
    
    output.append(str(sed.shape[0]*sed.shape[1])+"\n")  # nmod
    output.append(str(len(frequency))+"\n")  # nfreq (nwavelength)
    
    output.append("nu\n")  # type of independent variable (nu or lambda)
    output.append("1.0\n")  # conversion factor for independent variable
    output.append("F_nu\n")  # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
    output.append("1.0\n")  # conversion factor for dependent variable
    
    for z in Z:
        for a in ages:  # available SED ages
            output.append(str(a)+' '+str(z)+"\n")  # (npar x nmod) parameters
            
    output.append(' '.join(map(str,frequency))+"\n")  # the frequency(wavelength) grid, nfreq points
    
    
    for i,z in enumerate(Z):
        for j,a in enumerate(ages):  # available SED ages
            output.append(' '.join(map(str,sed[i,j]))+"\n")

    
    target.writelines(output)
    target.close()
    
    # ---- compile ascii file    i
    print('Compiling Cloudy atmosphere file (.ascii)')
    subprocess.call('echo -e \'compile stars \"'+wd+'Cloudy_input/'+model+'/'+model+'.ascii"\' | ~/c13.03/source/cloudy.exe',shell=True)

    # ---- copy .mod file to cloudy data directory
    print('Copying compiled atmosphere to Cloudy directory, '+cloudy_directory)
    subprocess.call('cp '+out_dir+model+'.mod '+cloudy_directory+'data/.', shell=True)

    # ---- remove .ascii file
    os.remove(out_dir+model+'.ascii')	
    
    ##
    ## Write input scripts
    ##
    
    print('Writing input scripts')
    
    ES_Z_sol = np.array([0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0])
    ES_helium = np.array([0.898,0.898,0.898,0.902,0.91,0.914,0.924,0.936,0.944,0.968,1.0,1.055,1.11])
    ES_hydrogen = np.array([1.06785,1.06785,1.06785,1.06514,1.05971,1.057,1.04993,1.04286,1.034,1.02143,1.0,0.9643,0.9286])
    
    # ----- index file parameters for array job execution
    filenames = []
    
    
    for i,z in enumerate(Z):
        
        for j,a in enumerate(ages):  # available SED ages       
            
            Z_sol = z/0.02
            
            # different value to Stephen as including the whole spectrum
            luminosity = np.log10(np.trapz(sed[i,j][::-1]/3.846E33, x=frequency[::-1])) + 6.
            
            
            output = []
        
            filename = str(i)+'_'+str(j)
            
            target = open(out_dir+filename+'.in','w')
            output.append('table star "'+model+'.mod" '+str(a)+' '+str(z)+'\n')
            
            
            filenames.append(filename+'\n')
            
            output.append('luminosity log solar '+str(luminosity)+'\n')
            output.append('metals '+str(Z_sol)+' linear deplete\n')
            output.append('element scale factor hydrogen '+str(np.interp(Z_sol, ES_Z_sol, ES_hydrogen))+' linear\n')
            output.append('element scale factor helium '+str(np.interp(Z_sol, ES_Z_sol, ES_helium))+' linear\n')
            output.append('hden 2 log constant density\n')
            output.append('sphere\n')
            output.append('covering factor 1.0 linear\n')
            output.append('radius 1 log parsec\n')
            output.append('iterate to convergence\n')
            output.append('set temperature floor 100 linear\n')
            output.append('stop temperature 100K\n')
            output.append('stop efrac -2\n')
            output.append('punch last continuum "'+cloudy_dir+str(i)+'_'+str(j).replace('.','')+'.cont"'+' units Angstroms no clobber\n')
            output.append('punch last lines, array "'+cloudy_dir+str(i)+'_'+str(j).replace('.','')+'.lines"'+' units Angstroms no clobber\n')
            
            
            target.writelines(output)
            target.close()
    
        
    print('Writing filenames')    

    target = open(out_dir+'filenames.txt','w')
    target.writelines(filenames)
    target.close()
    
    output = []    

    output.append('#$ -S /bin/bash\n')
    output.append('#$ -wd %sCloudy_input/%s\n'%(wd,model))
    output.append('#$ -j y\n')
    output.append('#$ -t 1-'+str(sed.shape[0]*sed.shape[1])+'\n')
    output.append('#$ -o '+cloudy_output_dir+'\n')
    output.append('#$ -N '+model+'\n')
    output.append('echo "Running job script"\n')
    output.append('INPUT_FILE=$(sed -n "$SGE_TASK_ID"p filenames.txt)\n')
    output.append('echo "$INPUT_FILE"\n')
    output.append('~/c13.03/source/cloudy.exe -r $INPUT_FILE\n')
    output.append('echo "Finished job script"\n')

    target = open(out_dir+'compile.job','w')
    target.writelines(output)
    target.close()

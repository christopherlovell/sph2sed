import numpy as np

def interp_coefficients(f, a, Z):
    return np.array([f[c]((a, Z)) for c in np.arange(len(f))]).flatten()

def reconstruct_spectra(mean, coeffs, components):
    return mean + np.dot(coeffs, components)


import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_spectra(wavelength, spectra):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,10))
    ax1.semilogx(wavelength, spectra)
    ax1.set_xlabel('$\AA$')
    ax1.set_ylabel('$\mathrm{log_{10}}(L_{\odot} \,/\, \AA)$')

    ax2.semilogx(wavelength, 10**spectra)
    ax2.set_xlabel('$\AA$')
    ax2.set_ylabel('$L_{\odot \,/\, \AA}$')

    figfile = BytesIO()
    plt.savefig(figfile, format='png', bbox_inches='tight')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue()).decode("utf-8")
    return figdata_png

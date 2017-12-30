from flask import Flask, render_template, request
from compute import interp_coefficients, reconstruct_spectra, plot_spectra
from model import InputForm

import pickle as pcl

app = Flask(__name__)

wavelength, mean_spectra, components = pcl.load(open('comp.p','rb'))
f = pcl.load(open('interp.p','rb'))

# View
@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        a = form.age.data
        Z = form.metal.data
        coeffs = interp_coefficients(f, a, Z)
        spectra = reconstruct_spectra(mean_spectra, coeffs, components)
        plot= plot_spectra(wavelength, spectra)
    else:
        coeffs = []
        plot = []
        spectra = []

    return render_template("index.html", form=form, coeffs=coeffs, spectra=spectra, plot=plot)

if __name__ == '__main__':
    app.run(debug=True)

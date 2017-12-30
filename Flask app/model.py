from wtforms import Form, FloatField, validators

class InputForm(Form):
    age = FloatField(label='Age', default=13.0,
                     validators=[validators.NumberRange(0, 13.7)])

    metal = FloatField(label='Metallicity', default=0.01,
                   validators=[validators.NumberRange(3e-3, 1.4e-2)])

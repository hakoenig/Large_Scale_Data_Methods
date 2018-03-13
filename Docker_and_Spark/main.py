"""Code for feature collection form."""

# Code adjusted from:
# https://pythonspot.com/flask-web-forms/
# Using bootstrap from:
# https://getbootstrap.com/docs/3.3/getting-started/

import json
import requests
from wtforms import Form, FloatField, validators
from flask import Flask, render_template, flash, request


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    """Class for Feature Input Form."""

    sepal_length = FloatField('Sepal Length in cm:',
                              validators=[validators.required()])
    sepal_width = FloatField('Sepal Width in cm:',
                             validators=[validators.required()])
    petal_length = FloatField('Petal Length in cm:',
                              validators=[validators.required()])
    petal_width = FloatField('Petal Width in cm:',
                             validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello():
    """Load site and make request."""
    form = ReusableForm(request.form)

    print(form.errors)
    if request.method == 'POST':
        dictToSend = {'sepal_length': request.form['sepal_length'],
                      'sepal_width': request.form['sepal_width'],
                      'petal_length': request.form['petal_length'],
                      'petal_width': request.form['petal_width']}
        print(dictToSend)
        res = requests.post('http://127.0.0.1:5001/get_prediction',
                            json=dictToSend)

        pred_dict = json.loads(res._content.decode('utf-8'))

        if form.validate():
            # Save the comment here.
            flash('Thanks for submitting your features!')
            return render_template("result.html", result=pred_dict)

        else:
            flash('Error: Some feature values were not correct/missing.')

    return render_template('hello.html', form=form)


if __name__ == "__main__":
    app.run()

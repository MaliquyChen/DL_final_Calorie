import flask
import os
from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import time
from inference import foodclassification

# UPLOAD_FOLDER = './uploaded_files'
UPLOAD_FOLDER = './uploaded_files'
ALLOWED_EXTENSIONS = {'jpg'}

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            food_names = foodclassification(filename)
            return food_names
            #return redirect(url_for('result', name=filename, action="POST"))

    return render_template('home.html')

'''
@app.route('/result/<action>/<name>')
def result(name, action):
    food_names = foodclassification(name)
    return food_names
    #return render_template('result.html')
'''

#rsplit: split strings to a list
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
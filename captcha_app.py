import urllib
import time, os
from captcha_recognizer import OCR_recognizer
from flask import Flask, render_template, request
app = Flask(__name__)
app.initialized = False

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/recognize", methods=['POST'])
def recognize():
    imageurl = request.form['imageurl']
    if imageurl:        
        urllib.urlretrieve(imageurl, 'images/recog.jpg')
        if not app.initialized:
            app.recognizer = OCR_recognizer('./models/sintegra_sc_model.ckpt')
            app.initialized = True
        recognized = app.recognizer.recognize('./images/recog.jpg')
        # recognizer = None
        # del recognizer
        return render_template('recognized.html', imageurl=imageurl,
                               recognized=recognized)

if __name__ == "__main__":
    app.run(debug=True)    

import urllib
import time, os
import captcha_crawl
from captcha_recognizer import OCR_recognizer
from flask import Flask, render_template, request, send_from_directory, make_response
app = Flask(__name__, static_url_path='')
app.initialized = False

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/recognize", methods=['POST'])
def recognize():
    imageurl = request.form['imageurl']
    image_fname = './images/recog.jpg'
    if imageurl:        
        urllib.urlretrieve(imageurl, 'images/recog.jpg')
    else:
        image_fname = './images/right.jpg'
    if not app.initialized:
        app.recognizer = OCR_recognizer('./models/sintegra_sc_model.ckpt')
        app.initialized = True
    print image_fname
    recognized = app.recognizer.recognize(image_fname)
    return render_template('recognized.html', imageurl=image_fname,
                               recognized=recognized)

@app.route('/images/<path:path>')
def send_images(path):
    response = make_response(send_from_directory('images', path))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/get_url')
def get_url_route():
    return captcha_crawl.get_url()

if __name__ == "__main__":
    app.run(debug=True)    

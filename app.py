from flask import Flask, render_template, request, jsonify
from module.crawling import crawl_cartoon
from module.detection import detect_main as detect
import cv2
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crawling_form')
def crawling_form():
    return render_template('crawl_cartoon_form.html')

@app.route('/crawling', methods=['POST'])
def crawling():
    data = request.form
    crawl_cartoon(data['url'], data['name'])
    return jsonify({'code':'200'})

@app.route('/detection_form')
def detection_form():
    return render_template('detection_form.html')

@app.route('/detection', methods=['POST'])
def detection():
    data = request.files.getlist('file')
    for image in data:
        extension = "." + image.filename.split('.')[-1]
        print(image.filename)
        if extension in ALLOWED_EXTENSIONS:
            image.save('./image/{0}'.format(secure_filename(image.filename)))
            cartoon_img = cv2.imread('./image/{0}'.format(secure_filename(image.filename)))
            img_detected = detect(cartoon_img)
            cv2.imwrite('./detected_image/{0}'.format(secure_filename(image.filename)), img_detected)
        else:
            return jsonify({'code':'400', 'message':"Not allowed extension"})
    return jsonify({'code':'200'})

if __name__ == '__main__':
    app.run(debug=True)
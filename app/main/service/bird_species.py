from flask import Blueprint, jsonify, request, current_app
from PIL import Image
import requests
import numpy as np
from io import BytesIO
from service.inference import init_model, inference

# define the blueprint
bird_species = Blueprint(name="bird_species", import_name=__name__)
model = init_model(current_app.config['model_path'])

# add view function to the blueprint
@bird_species.route('/test', methods=['GET'])
def test():
    output = {"msg": "I'm the test endpoint from blueprint_x."}
    return jsonify(output)

# add view function to the blueprint
# test url http://172.26.36.249:51480/api/v1/bird/image?url=https%3A%2F%2Fdrive.google.com%2Fuc%3Fexport%3Ddownload%26id%3D1Bt8BJsUmG7JMChPGTrp2XiQGqp0CTYl4
@bird_species.route('/image', methods=['POST','GET'])
def image_url():
    image_url = request.args.get('url')
    img_ret = {}
    img_size = 150
    images = []
    if image_url != None:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((img_size, img_size))
        images.append(img)
    images = np.array([np.array(img) for img in images])
    images = images / 255.0
    ret = inference(images, model)
    img_ret[image_url] = ret
    return jsonify(img_ret)
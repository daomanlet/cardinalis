import numpy as np
from tensorflow import keras
from service.label import LABELS
from PIL import Image

def init_model(model_path='./deploy/mobilenet_v2.h5'):
    model = None
    try:
        print(model_path)
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"An exception occurred: {e}")
    return model 

def inference(images, model):
    predictions = model.predict(images)
    ret = []
    for i in range(len(images)):
        predicted_class = np.argmax(predictions[i])
       # class_probability = predictions[i, predicted_class]
        ret.append(LABELS[predicted_class])
    return ret

def read_images_from_url(urls):
    images = []
    img_size = 150
    for filename in urls:
        img = Image.open(filename)
        img = img.resize((img_size, img_size))
        images.append(img)

    images = np.array([np.array(img) for img in images])
    images = images / 255.0
    return images

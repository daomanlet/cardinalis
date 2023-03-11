import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

def create_labels():
    train_dir = './models/training_data/train'
    batch_size = 32
    img_size = 150
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                       zoom_range=0.2,
                                       width_shift_range=0.2, height_shift_range=0.2
                                       )
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(
                                                            img_size, img_size),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')
    labels = [k for k in train_generator.class_indices]
    with open('./samples/label.py','w') as tfile:
        tfile.write('LABELS=[')
        tfile.write('",\n'.join(labels))

def invoke_model():
    from label import LABELS
    img_size = 150
    # Test local files
    image_directory = './models/mobilenet_v2/verify_images'
    model = keras.models.load_model('./deploy/mobilenet_v2.h5')
    image_directory = './models/mobilenet_v2/verify_images'
    images = []
    for filename in os.listdir(image_directory):
        path = os.path.join(image_directory, filename)
        img = Image.open(path)
        img = img.resize((img_size, img_size))
        images.append(img)

    images = np.array([np.array(img) for img in images])
    images = images / 255.0
    predictions = model.predict(images)
    for i in range(len(images)):
        predicted_class = np.argmax(predictions[i])
        class_probability = predictions[i, predicted_class]
        print(f'Predicted class for {i+1}.jpg : {LABELS[predicted_class]}')
        print('Class probability:', class_probability)


if __name__ == "__main__":
    invoke_model()

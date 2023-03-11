import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from PIL import Image
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow import keras


def train_model():
    train_dir = './models/training_data/train'
    val_dir = './models/training_data/valid'
    test_dir = './models/training_data/test'
    batch_size = 32
    img_size = 150
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                zoom_range=0.2,
                                width_shift_range=0.2,height_shift_range=0.2
                                )
    val_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen = ImageDataGenerator(rescale=1/255.)
    train_generator = train_datagen.flow_from_directory(train_dir,  
                                                    target_size=(img_size, img_size), 
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')  
    validation_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=(img_size, img_size),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(img_size, img_size),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='categorical')
    base_model = MobileNetV2(include_top = False,
                        weights = 'imagenet',
                        input_shape = (img_size,img_size,3))

    base_model.summary()

    num_layers = len(base_model.layers)
    num_layers

    for layer in base_model.layers[:num_layers//2]:
        layer.trainable = False

    base_model.summary()

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) # to prevent overfitting
    model.add(Dense(500, activation='softmax'))

    # start to train model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=10,
                        validation_data=validation_generator,
                        
                    )
    model.save('./models/finetuned_model/')

def batch_verify():
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                zoom_range=0.2,
                                width_shift_range=0.2,height_shift_range=0.2
                                )
    train_dir = './models/training_data/train'
    batch_size = 32
    img_size = 150
    train_generator = train_datagen.flow_from_directory(train_dir,  
                                                    target_size=(img_size, img_size), 
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')
    labels = [k for k in train_generator.class_indices]
    #Test local files
    image_directory = './models/mobilenet_v2/verify_images'
    model = keras.models.load_model('./models/finetuned_model/')
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
        print(f'Predicted class for {i+1}.jpg : {labels[predicted_class]}')
        print('Class probability:', class_probability)

def convert_model():
    model_chp = keras.models.load_model('./models/finetuned_model/')
    model_chp.save('./deploy/mobilenet_v2.h5')


if __name__ == "__main__":
    print('start')
    #convert_model()
    # batch_verify()
    # train_model()

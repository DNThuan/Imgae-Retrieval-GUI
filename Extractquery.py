from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling2D, Flatten



def extract_feature(image):

    input_shape = (224, 224, 3)

    # base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # model = Sequential([
    #                 # input_layers,
    #                 base_model,
    #                 MaxPooling2D(strides=(7, 7)),
    #                 Flatten()
    # ])


    base_model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs = base_model.input, outputs = base_model.layers[-2].output)

    
    img = Image.fromarray(image)
    img = img.resize(input_shape[: 2])
    img = tf.expand_dims(tf.convert_to_tensor(img), axis=0)
    feature = model.predict(img, batch_size=None, verbose=True, steps =1).squeeze()

    return feature

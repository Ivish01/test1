import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, Input, Dense, Reshape, Activation,UpSampling2D


base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               weights='imagenet',
                                               input_tensor=None, 
                                               input_shape=(256,256,3), 
                                               pooling=None,
                                               classes=1000,
                                               classifier_activation='softmax')


base_model.trainable = False



def vgg16_encoder_decoder(args):

    inputs = Input(shape=(args.img_height, args.img_width, args.nchannel))

    encoder = base_model(inputs)

    encoder_dim = K.int_shape(encoder)

    encoder = Flatten()(encoder)

    latent_space = Dense(args.latent_dim, name='latent_space')(encoder)

    decoder = Dense(np.prod(encoder_dim[1:]))(latent_space)
    decoder = Reshape((encoder_dim[1], encoder_dim[2], encoder_dim[3]))(decoder)

    decoder = UpSampling2D(size = (2,2))(decoder)
    decoder = Conv2D(512, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(512, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(512, 3, activation = 'relu', padding = 'same')(decoder)

    decoder = UpSampling2D(size = (2,2))(decoder)
    decoder = Conv2D(512, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(512, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(512, 3, activation = 'relu', padding = 'same')(decoder)
  

    decoder = UpSampling2D(size = (2,2))(decoder)
    decoder = Conv2D(256, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(256, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(256, 3, activation = 'relu', padding = 'same')(decoder)

    decoder = UpSampling2D(size = (2,2))(decoder)
    decoder = Conv2D(128, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(128, 3, activation = 'relu', padding = 'same')(decoder)

    decoder = UpSampling2D(size = (2,2))(decoder)
    decoder = Conv2D(64, 3, activation = 'relu', padding = 'same')(decoder)
    decoder = Conv2D(64, 3, activation = 'relu', padding = 'same')(decoder)

    decoder = Conv2DTranspose(3, (3, 3), padding="same")(decoder)

    output = Activation('sigmoid', name='decoder')(decoder)

    autoencoder_vgg16 = Model(inputs = inputs, outputs = output, name = 'autoencoder_vgg16')
    return autoencoder_vgg16


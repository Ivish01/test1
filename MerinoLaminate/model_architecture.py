import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Dense, Reshape, Activation






def encoder_decoder_model(args):
    
    input_model = Input(shape=(args.img_height, args.img_width, args.nchannel))

    # Encoder layers
    encoder = Conv2D(32, (3,3), padding='same', kernel_initializer='normal')(input_model)
    encoder = LeakyReLU()(encoder)
    encoder = BatchNormalization(axis=-1)(encoder)

    encoder = Conv2D(64, (3,3), padding='same', kernel_initializer='normal')(encoder)
    encoder = LeakyReLU()(encoder)
    encoder = BatchNormalization(axis=-1)(encoder)

    encoder = Conv2D(64, (3,3), padding='same', kernel_initializer='normal')(input_model)
    encoder = LeakyReLU()(encoder)
    encoder = BatchNormalization(axis=-1)(encoder)

    encoder_dim = K.int_shape(encoder)
    encoder = Flatten()(encoder)

    # Latent Space
    latent_space = Dense(args.latent_dim, name='latent_space')(encoder)

    # Decoder Layers
    decoder = Dense(np.prod(encoder_dim[1:]))(latent_space)
    decoder = Reshape((encoder_dim[1], encoder_dim[2], encoder_dim[3]))(decoder)

    decoder = Conv2DTranspose(64, (3,3), padding='same', kernel_initializer='normal')(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = BatchNormalization(axis=-1)(decoder)

    decoder = Conv2DTranspose(64, (3,3), padding='same', kernel_initializer='normal')(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = BatchNormalization(axis=-1)(decoder)

    decoder = Conv2DTranspose(32, (3,3), padding='same', kernel_initializer='normal')(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = BatchNormalization(axis=-1)(decoder)

    decoder = Conv2DTranspose(3, (3, 3), padding="same")(decoder)
    output = Activation('sigmoid', name='decoder')(decoder)

    # Create model object
    autoencoder = Model(input_model, output, name='autoencoder')
    
    return autoencoder



import Data_input
import model_architecture


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


#loading model Architecture
Autoencoder = model_architecture.encoder_decoder_model()


#loading Training and Validation Data
batch_size = 1
training_set = Data_input.load_train("data/training", 256, 256, 1)
validation_set = Data_input.load_val("data/validation", 256, 256, 1)

Autoencoder.compile(loss="mse", optimizer= Adam(learning_rate=1e-3))

# Fit the model
history = Autoencoder.fit_generator(
          training_set,
          steps_per_epoch=training_set.n // batch_size,
          epochs=10,
          validation_data=validation_set,
          validation_steps=validation_set.n // batch_size,
          callbacks = [ModelCheckpoint('models/image_autoencoder_2.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)])






                                                     
                                                     
                                                     
    
    
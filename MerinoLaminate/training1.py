import argparse

import Data_input
import model_architecture
import model_architecture_vgg16

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt




parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_dir',      type=str,   default='data1/training',        help='input train directory (in train subfolder)')
parser.add_argument('--val_dir',        type=str,   default='data1/validation',      help='input train directory (in train subfolder)')
parser.add_argument('--img_width',      type=int,   default= 256,                    help='input image width')
parser.add_argument('--img_height',     type=int,   default= 256,                    help='input image height')
parser.add_argument('--batch_size',     type=int,   default=1,                       help='batch size')
parser.add_argument('--nchannel',       type=int,   default=3,                       help='image channels')
parser.add_argument('--latent_dim',     type=int,   default=16,                      help='latent dimension')
parser.add_argument('--model',          type=str,   default='vgg16',                 help='vanilla_network or vgg16')
parser.add_argument('--epochs',         type=int,   default=2,                       help='training epochs')
parser.add_argument('--learn_rate',     type=float, default=0.001,                   help='learning rate')




args = parser.parse_args()


def train(args):
    
    training_set = Data_input.load_train(args)
    validation_set = Data_input.load_val(args)
    
    Autoencoder = model_architecture.encoder_decoder_model(args)
    Autoencoder_vgg = model_architecture_vgg16.vgg16_encoder_decoder(args)
    
    
    

   # Fit the model
    if args.model == 'vanilla_network':
        
        Autoencoder.compile(loss="mse", optimizer= Adam(learning_rate=args.learn_rate))
       
   
        history = Autoencoder.fit_generator(training_set,
                                  steps_per_epoch=training_set.n // args.batch_size,
                                  epochs=args.epochs,
                                  validation_data=validation_set,
                                  validation_steps=validation_set.n // args.batch_size,
                                  callbacks = [ModelCheckpoint('models/image_autoencoder_vanilla.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)])
        
        
    if args.model == 'vgg16':
        
        Autoencoder_vgg.compile(loss="mse", optimizer= Adam(learning_rate=args.learn_rate))
        
        history = Autoencoder_vgg.fit_generator(training_set,
                                  steps_per_epoch=training_set.n // args.batch_size,
                                  epochs=args.epochs,
                                  validation_data=validation_set,
                                  validation_steps=validation_set.n // args.batch_size,
                                  callbacks = [ModelCheckpoint('models/image_autoencoder_vgg.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)])
        
        
    
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    Epochs = range(len(train_loss))
 
    plt.figure()
 
    plt.plot(Epochs, train_loss, 'b', label='Training Loss')
    plt.plot(Epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("train_val_loss.jpg")
    plt.show()
    
    
    
    return train_loss, val_loss



if __name__ == "__main__":
    train(args)






                                                     
                                                     
                                                     
    
    
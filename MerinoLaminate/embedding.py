
import pickle
from tensorflow.keras.models import Model, load_model
import os
from tensorflow.keras.preprocessing.image import  load_img, img_to_array
import numpy as np



import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_dir',      type=str,   default='models/image_autoencoder_vgg.h5',        help='input model directory')
parser.add_argument('--image_dir',      type=str,   default='data1/training/laminates',                help='input image directory for embeddings)')
parser.add_argument('--img_width',      type=int,   default= 256,                                     help='input image width')
parser.add_argument('--img_height',     type=int,   default= 256,                                     help='input image height')
parser.add_argument('--latent_dim',     type=int,   default=16,                                       help='latent dimension')


args = parser.parse_args()


def create_embedding(args):
    autoencoder = load_model(args.model_dir, compile=False)
    latent_space_model = Model(autoencoder.input, autoencoder.get_layer('latent_space').output)
    
    
    X = []
    indices = []

    for i in range(len(os.listdir(args.image_dir))):
        try:
            img_name = os.listdir(args.image_dir)[i]
            img = load_img(args.image_dir+'/{}'.format(img_name), target_size = (args.img_height, args.img_width))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            pred = latent_space_model.predict(img)
            pred = np.resize(pred, (args.latent_dim))
            X.append(pred)
            indices.append(img_name)

        except:
            print(img_name)
            
            
    embeddings = {'indices': indices, 'features': np.array(X)}

    pickle.dump(embeddings, open('models/image_embeddings_2.pickle', 'wb'))
    
    return print("no of emdeddings  ", len(embeddings))

if __name__ == "__main__":
    
    create_embedding(args)

    
    
    
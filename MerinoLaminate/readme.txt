Merinolens:


There are the scripts for training and embeddings:

Data_input: preprocessing of the input images for model training.

model_architecture : vanilla encoder-decoder network for training  image similarity.

model_architecture_vgg :  pretrained vgg16 network as a encoder and created decoder network.

training: main file to be run for training which imported(Data_input and model_architecture_vgg)

emdedding: ones the model is trained, extracted latent layer , and created embedding for the images.

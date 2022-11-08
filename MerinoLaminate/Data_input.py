
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def load_train(args):
    train_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(args.train_dir,
                                                     target_size = (args.img_height, args.img_width),
                                                     batch_size = args.batch_size,
                                                     class_mode = 'input',
                                                     shuffle=True)
    return training_set
                                                     

def load_val(args):
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_set = train_datagen.flow_from_directory(args.val_dir,
                                                     target_size = (args.img_height, args.img_width),
                                                     batch_size = args.batch_size,
                                                     class_mode = 'input',
                                                     shuffle=True)
    return validation_set


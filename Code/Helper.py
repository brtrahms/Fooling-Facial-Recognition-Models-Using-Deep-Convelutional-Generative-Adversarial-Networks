import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from keras.utils.data_utils import get_file

V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'
VGGFACE_DIR = 'models/vggface'

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*4112, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 4112)))
    assert model.output_shape == (None, 7, 7, 4112)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(2056, (2, 2), strides = (2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 2056)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1028, (5, 5), strides = (2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 1028)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(512, (5, 5), strides = (1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides = (2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides = (2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 112, 112, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides = (1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 112, 112, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(3, (5, 5), strides = (2,2), padding='same', use_bias=False, activation = 'sigmoid'))
    assert model.output_shape == (None, 224, 224, 3)

    return model

def generator_loss(output, ID):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    target = tf.ones_like(output[...,0])
    return cross_entropy(target, output[..., ID[0][0]]) 

def train(generator, model, gen_opt, BATCH, i, loss, Target_ID):
        
    noise = tf.random.normal([BATCH,100])
    
    with tf.GradientTape() as gen_tape:
        generated_image = generator(noise.numpy(), training=True)
        
        generated_image = generated_image * 255
        generated_image = generated_image[..., ::-1] - [91.4953, 103.8827, 131.0912]
        
        yhat = model(generated_image)

        #Loss array
        gen_loss = generator_loss(yhat, Target_ID)

    loss[0].append(i)
    loss[1].append(gen_loss)
    
    if(i % 50 == 0):
        print("Loss at Epoch " + str(i) + " is " + str(gen_loss.numpy()))
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
def decode_predictions(preds, top=5):
    LABELS = None
    if len(preds.shape) == 2:
        if preds.shape[1] == 2622:
            fpath = get_file('rcmalli_vggface_labels_v1.npy',
                             V1_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
            LABELS = np.load(fpath)
        elif preds.shape[1] == 8631:
            fpath = get_file('rcmalli_vggface_labels_v2.npy',
                             V2_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
            LABELS = np.load(fpath)
        else:
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                             '(samples, 8631) for V2.'
                             'Found array with shape: ' + str(preds.shape))
    else:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                         '(samples, 8631) for V2.'
                         'Found array with shape: ' + str(preds.shape))
    results = []
        
    for pred in preds:
        top_indices = tf.argsort(pred)[-top:][::-1]
        result = [[str(LABELS[i].encode('utf8')), pred[i], i] for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results
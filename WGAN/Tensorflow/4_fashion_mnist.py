import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import pickle
import tensorflow as tf
import argparse
from keras.models import load_model

import tflib.mnist
from classes.models_wgan import MnistWganInv
from classes.search import iterative_search, recursive_search


# classifier
classifier_rf = pickle.load(open('./models/mnist_rf_9045.sav', 'rb'))

def cla_fn_rf(x):
    return classifier_rf.predict(np.reshape(x, (-1, 784)))

graph_CLA = tf.Graph()
with graph_CLA.as_default():
    classifier_nn = load_model("./models/mnist_lenet_9871.h5")

def cla_fn_nn(x):
    with graph_CLA.as_default():
        return np.argmax(classifier_nn.predict_on_batch(np.reshape(x, (-1, 1, 28, 28))), axis=1)


# Initialization of the WGAN and of the Inverter
graph_GAN = tf.Graph()
with graph_GAN.as_default():
    sess_GAN = tf.Session()
    model_GAN = MnistWganInv()
    saver_GAN = tf.train.Saver(max_to_keep=100)
    saver_GAN = tf.train.import_meta_graph('{}.meta'.format('./models/model-47999'))
    saver_GAN.restore(sess_GAN, './models/model-47999')

# Initialization of the Generator
def gen_fn(z):
    with sess_GAN.as_default():
        with graph_GAN.as_default():
            # x_prime = G(z)
            x_p = sess_GAN.run(model_GAN.generate(tf.cast(tf.constant(np.asarray(z)), 'float32')))
    return x_p

# Initialization of the Inverter
def inv_fn(x):
    with sess_GAN.as_default():
        with graph_GAN.as_default():
            # z_prime = I(x)
            z_p = sess_GAN.run(model_GAN.invert(x))
    return z_p

search = recursive_search

# MNIST fashion data
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = tf.keras.datasets.fashion_mnist.load_data()

output_delta_z = np.ndarray(x_test_fashion.shape[0])

for i in range(x_test_fashion.shape[0]):
    x = x_test_fashion[i, :]

    # getting the prediction of the black box classifier
    y_pred = cla_fn_nn(x)[0]

    # generating the natural adversary example
    adversary = search(gen_fn, inv_fn, cla_fn_nn, x, y_pred,
                       nsamples=5000, step=0.01, verbose=False)

    output_delta_z[i] = adversary["delta_z"]
    print("delta_z of point", i, " :", output_delta_z[i])

np.save("deltaz_mnist_fashion_mnist.npy", output_delta_z)

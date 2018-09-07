import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

import pickle
import tensorflow as tf
import argparse
from keras.models import load_model

import tflib.mnist
from mnist_wgan_inv import MnistWganInv
from search import iterative_search, recursive_search


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


### Chicken example
ck = io.imread('chicken.jpg', as_gray=True)
ck = resize(ck, (28, 28), anti_aliasing = True)
ck = np.float32(ck)

ck_data = ck.reshape(1, 28*28)

# RF prediction
y_hat_rf = cla_fn_rf(ck)
print("estimate for the chicken (random forest)", y_hat_rf)
ck_probabilities_rf = classifier_rf.predict_proba(ck_data)
print("estimated probabilities for the chicken (random forest)", ck_probabilities_rf)

# NN prediction
y_hat_nn = cla_fn_nn(ck)
print("estimate for the chicken (LeNet)", y_hat_nn)
ck_probabilities_nn = classifier_nn.predict_proba( np.reshape(ck_data, (-1, 1, 28, 28)) )
print("estimated probabilities for the chicken (LeNet)", ck_probabilities_nn)

adversary_ck_rf = search(gen_fn, inv_fn, cla_fn_rf, ck_data, y_hat_rf,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the chicken (random forest):", adversary_ck_rf["delta_z"])

adversary_ck_nn = search(gen_fn, inv_fn, cla_fn_nn, ck_data, y_hat_nn,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the chicken (neural network):", adversary_ck_nn["delta_z"])


### Let's try with a falafel picture
fl = io.imread('falafel.png', as_gray=True)
fl = resize(fl, (28, 28), anti_aliasing = True)
fl = np.float32(fl)
fl_data = fl.reshape(1, 28*28)

# RF prediction
y_hat_rf = cla_fn_rf(fl)
print("estimate for the falafel (random forest)", y_hat_rf)
fl_probabilities_rf = classifier_rf.predict_proba(fl_data)
print("estimated probabilities for the falafel (random forest)", fl_probabilities_rf)

# NN prediction
y_hat_nn = cla_fn_nn(fl)
print("estimate for the falafel (LeNet)", y_hat_nn)
fl_probabilities_nn = classifier_nn.predict_proba( np.reshape(fl_data, (-1, 1, 28, 28)) )
print("estimated probabilities for the falafel (LeNet)", fl_probabilities_nn)

adversary_fl_rf = search(gen_fn, inv_fn, cla_fn_rf, fl_data, y_hat_rf,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the falafel (random forest):", adversary_fl_rf["delta_z"])

adversary_fl_nn = search(gen_fn, inv_fn, cla_fn_nn, fl_data, y_hat_nn,
                   nsamples=5000, step=0.01, verbose=False)

print("delta_z for the falafel (neural network):", adversary_fl_nn["delta_z"])

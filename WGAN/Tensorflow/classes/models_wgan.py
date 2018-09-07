import os, sys, time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import numpy as np
import tensorflow as tf
import argparse

import tflib
import tflib.mnist
import tflib.plot
import tflib.save_images
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear


class MnistWganInv(object):
    def __init__(self, x_dim=784, z_dim=64, latent_dim=64, batch_size=80,
                 c_gp_x=10., lamda=0.1, output_path='./'):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        # Gradient penalty (lambda in the improved WGAN training paper)
        self.c_gp_x = c_gp_x
        # Lambda parameter for the Inverter
        self.lamda = lamda
        self.output_path = output_path

        self.gen_params = self.dis_params = self.inv_params = None

        # z, original noise
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        # generated x from noise, x^prime = G(z)
        self.x_p = self.generate(self.z)

        # x, original data
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])

        # z^prime = I(x), inverted x (dense representation of x)
        self.z_p = self.invert(self.x)

        # Discriminator on original x
        self.dis_x = self.discriminate(self.x)

        # Discriminator on dense representation of x
        self.dis_x_p = self.discriminate(self.x_p)

        # reconstructed x, G(z^prime) = G(I(x)), using the Generator on the dense representation
        # of the original x
        self.rec_x = self.generate(self.z_p)

        # reconstructed z, I(x^prime) = I(G(z)), using the Inverter on the x generated from z
        self.rec_z = self.invert(self.x_p)

        # loss function for the Generator training
        self.gen_cost = -tf.reduce_mean(self.dis_x_p)

        # loss function for the Inverter training
        self.inv_cost = tf.reduce_mean(tf.square(self.x - self.rec_x))
        self.inv_cost += self.lamda * tf.reduce_mean(tf.square(self.z - self.rec_z))

        # loss function for the Discriminator training
        self.dis_cost = tf.reduce_mean(self.dis_x_p) - tf.reduce_mean(self.dis_x)

        # TODO: check in details this part
        # These are the parameters of the "Improved Training of Wasserstein GANs"
        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
        difference = self.x_p - self.x
        interpolate = self.x + alpha * difference
        gradient = tf.gradients(self.discriminate(interpolate), [interpolate])[0]
        slope = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
        gradient_penalty = tf.reduce_mean((slope - 1.) ** 2)
        self.dis_cost += self.c_gp_x * gradient_penalty

        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.gen_cost, var_list=self.gen_params)
        self.inv_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.inv_cost, var_list=self.inv_params)
        self.dis_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.dis_cost, var_list=self.dis_params)

    def generate(self, z):
        assert z.shape[1] == self.z_dim

        output = tflib.ops.linear.Linear('Generator.Input', self.z_dim,
                                         self.latent_dim * 64, z)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, self.latent_dim * 4, 4, 4])  # 4 x 4

        output = tflib.ops.deconv2d.Deconv2D('Generator.2', self.latent_dim * 4,
                                             self.latent_dim * 2, 5, output)
        output = tf.nn.relu(output)  # 8 x 8
        output = output[:, :, :7, :7]  # 7 x 7

        output = tflib.ops.deconv2d.Deconv2D('Generator.3', self.latent_dim * 2,
                                             self.latent_dim, 5, output)
        output = tf.nn.relu(output)  # 14 x 14

        output = tflib.ops.deconv2d.Deconv2D('Generator.Output',
                                             self.latent_dim, 1, 5, output)
        output = tf.nn.sigmoid(output)  # 28 x 28

        if self.gen_params is None:
            self.gen_params = tflib.params_with_name('Generator')

        return tf.reshape(output, [-1, self.x_dim])

    def discriminate(self, x):
        output = tf.reshape(x, [-1, 1, 28, 28])  # 28 x 28

        output = tflib.ops.conv2d.Conv2D(
            'Discriminator.Input', 1, self.latent_dim, 5, output, stride=2)
        output = tf.nn.leaky_relu(output)  # 14 x 14

        output = tflib.ops.conv2d.Conv2D(
            'Discriminator.2', self.latent_dim, self.latent_dim * 2, 5,
            output, stride=2)
        output = tf.nn.leaky_relu(output)  # 7 x 7

        output = tflib.ops.conv2d.Conv2D(
            'Discriminator.3', self.latent_dim * 2, self.latent_dim * 4, 5,
            output, stride=2)
        output = tf.nn.leaky_relu(output)  # 4 x 4
        output = tf.reshape(output, [-1, self.latent_dim * 64])

        output = tflib.ops.linear.Linear(
            'Discriminator.Output', self.latent_dim * 64, 1, output)
        output = tf.reshape(output, [-1])

        if self.dis_params is None:
            self.dis_params = tflib.params_with_name('Discriminator')

        return output

    def invert(self, x):
        output = tf.reshape(x, [-1, 1, 28, 28])  # 28 x 28

        output = tflib.ops.conv2d.Conv2D(
            'Inverter.Input', 1, self.latent_dim, 5, output, stride=2)
        output = tf.nn.leaky_relu(output)  # 14 x 14

        output = tflib.ops.conv2d.Conv2D(
            'Inverter.2', self.latent_dim, self.latent_dim * 2, 5, output,
            stride=2)
        output = tf.nn.leaky_relu(output)  # 7 x 7

        output = tflib.ops.conv2d.Conv2D(
            'Inverter.3', self.latent_dim * 2, self.latent_dim * 4, 5,
            output, stride=2)
        output = tf.nn.leaky_relu(output)  # 4 x 4
        output = tf.reshape(output, [-1, self.latent_dim * 64])

        output = tflib.ops.linear.Linear(
            'Inverter.4', self.latent_dim * 64, self.latent_dim * 8, output)
        output = tf.nn.leaky_relu(output)

        output = tflib.ops.linear.Linear(
            'Inverter.Output', self.latent_dim * 8, self.z_dim, output)
        output = tf.reshape(output, [-1, self.z_dim])

        if self.inv_params is None:
            self.inv_params = tflib.params_with_name('Inverter')

        return output

    def train_gen(self, sess, x, z):
        _gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _gen_cost

    def train_dis(self, sess, x, z):
        _dis_cost, _ = sess.run([self.dis_cost, self.dis_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _dis_cost

    def train_inv(self, sess, x, z):
        _inv_cost, _ = sess.run([self.inv_cost, self.inv_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        # x_p = G(z)
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        tflib.save_images.save_images(
            samples.reshape((-1, 28, 28)),
            os.path.join(self.output_path, 'examples/samples_{}.png'.format(frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        # rec_x = G(z_p) = G(I(x))
        reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})
        comparison = np.zeros((images.shape[0] * 2, images.shape[1]),
                              dtype=np.float32)
        for i in xrange(images.shape[0]):
            comparison[2 * i] = images[i]
            comparison[2 * i + 1] = reconstructions[i]
        tflib.save_images.save_images(
            comparison.reshape((-1, 28, 28)),
            os.path.join(self.output_path, 'examples/recs_{}.png'.format(frame)))
        return comparison

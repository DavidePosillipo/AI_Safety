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
import classes.tflib.mnist
import classes.tflib.plot
import classes.tflib.save_images
import classes.tflib.ops.batchnorm
import classes.tflib.ops.conv2d
import classes.tflib.ops.deconv2d
import classes.tflib.ops.linear

from classes.models_wgan import MnistWganInv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--z_dim', type=int, default=64, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=5,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=.1,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    args = parser.parse_args()


    # dataset iterator
    # train: training set
    # dev: development set (i.e. validation set)
    # test: test set
    train_gen, dev_gen, test_gen = tflib.mnist.load(args.batch_size, args.batch_size)

    def inf_train_gen():
        while True:
            for instances, labels in train_gen():
                yield instances

    _, _, test_data = tflib.mnist.load_data()
    fixed_images = test_data[0][:32]
    del test_data

    tf.set_random_seed(326)
    np.random.seed(326)
    fixed_noise = np.random.randn(64, args.z_dim)

    mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, latent_dim=args.latent_dim,
        batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path)

    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        # loop over the overall iterations (default = 100.000)
        for iteration in range(args.iterations):
            # loop over discriminator_steps (default = 5)
            for i in range(args.dis_iter):
                noise = np.random.randn(args.batch_size, args.z_dim)
                images = inf_train_gen().next()
                # list with losses of the discriminator
                dis_cost_lst += [mnistWganInv.train_dis(session, images, noise)]
                # list with losses of the inverter
                inv_cost_lst += [mnistWganInv.train_inv(session, images, noise)]
            # end of the loop over discriminator_steps

            # loss of the generator
            gen_cost = mnistWganInv.train_gen(session, images, noise)
            # loss of the discriminator as the mean of the dis_iter losses
            dis_cost = np.mean(dis_cost_lst)
            # loss of the inverter as the mean of the dis_iter losses
            inv_cost = np.mean(inv_cost_lst)

            tflib.plot.plot('train gen cost', gen_cost)
            tflib.plot.plot('train dis cost', dis_cost)
            tflib.plot.plot('train inv cost', inv_cost)

            # with iterations=100.000, at the iteration 99, 999, 9999, 99999
            # For these iterations, apply the generator over some noise examples and
            # the G(I(.)) reconstructions over some images
            if iteration % 100 == 99:
                mnistWganInv.generate_from_noise(session, fixed_noise, iteration)
                mnistWganInv.reconstruct_images(session, fixed_images, iteration)

            # At the 999, 9.999, 99.999 iterations, save the model
            if iteration % 1000 == 999:
                save_path = saver.save(session, os.path.join(
                    args.output_path, 'models/model'), global_step=iteration)

            # At the 999, 9.999, 99.999 iterations, evaluate the discriminator and
            # the inverter using the development data set
            if iteration % 1000 == 999:
                dev_dis_cost_lst, dev_inv_cost_lst = [], []
                for dev_images, _ in dev_gen():
                    noise = np.random.randn(args.batch_size, args.z_dim)
                    dev_dis_cost, dev_inv_cost = session.run(
                        [mnistWganInv.dis_cost, mnistWganInv.inv_cost],
                        feed_dict={mnistWganInv.x: dev_images,
                                   mnistWganInv.z: noise})
                    dev_dis_cost_lst += [dev_dis_cost]
                    dev_inv_cost_lst += [dev_inv_cost]
                tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
                tflib.plot.plot('dev inv cost', np.mean(dev_inv_cost_lst))

            if iteration < 5 or iteration % 100 == 99:
                tflib.plot.flush(os.path.join(args.output_path, 'models'))

            tflib.plot.tick()
        # end of the loop over the overall iterations

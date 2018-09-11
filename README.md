# Safety in AI
## Robustness metrics for Deep Learning predictions
Portfolio Project for the Data Science Retreat

**WORK IN PROGRESS: THE REPO WILL BE UPDATED IN THE NEXT DAYS. Thanks for the patience.**

How reliable are predictions of Deep Learning algorithms? In this project I investigated:
1. an extension of WGAN (Wasserstein Genetarive Adversarial Networks);
2. an application of VAE (Variational Autoencoder)

in order to produce a meaningful measure of robustness.

This approaches are applicable to every black box classifier and they don't need any modification of the used classifier.

Using this approach is possible to create an alarm system that alerts the user if an online prediction is weak. This is particularly important in scenarios where AI could affect the safety of people or environment.

Main idea of both the approaches:
1. learn how the training set data are made in a “deep” level (latent representation)
2. check if your new data are “too” different respect to the training data, in the latent representation.

If the new data are "too" different, than you should consider as not completely reliable your prediction (or avoid it).


### Main results
A summary of the results will be added here soon. Now you can give a look to the slides (AI_Safety_presentation_community_day.pdf) that I used for the Data Science Retreat Community Day presentation.

### Usage
To set up the environment, do:
```
pip install -r requirements.txt
```

#### WGAN + Inverter
At the moment, the WGAN approach is usable only with a GPU.

To test the WGAN, go to the WGAN folder and:
```
python 1_training.py
```
for training the WGAN + Inverter.

```
python 2_distribution_delta_z.py
```
in order to get the distribution of delta_z for the MNIST test set.

```
python 3_chicken.py
```
to test the chicken picture.

```
python 4_fashion_mnist.py
```
to get the distribution of delta_z for the Fashion MNIST test set.

#### VAE
The VAE can be tested with CPU or GPU.

To test the VAE, go to the VAE folder and:
```
python 1_training.py
```
to train the VAE.

```
python 2_distribution_losses.py
```
in order to get the distribution of the VAE losses for the MNIST test set.

```
python 3_chicken.py
```
to test the chicken picture.

```
python 4_fashion_mnist.py
```
to get the distribution of the VAE losses for the Fashion MNIST test set.

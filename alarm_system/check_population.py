# Checking the population of the input, using the VAE approach
def check_population_VAE(input, threshold, VAE, loss_function):
    # Getting the reconstructed input and the parameters of the VAE
    recon, mu, logvar = VAE(input)
    # Computing the VAE loss for the input
    input_loss = loss_function(recon, input, mu, logvar)
    # If the input loss is bigger than the threshold, we guess that the point is
    # not coming from the known population
    if input_loss > threshold:
        return 0
    else:
        return 1

# Checking the population of the input, using the WGAN+Inverter approach 
def check_population_WGAN(input, threshold, generator, inverter, classifier, searcher):
    # Getting the prediction of the blak box classifier for the input
    y_pred = classifier(input)
    # Getting the delta_z for the input, using generator, inverter and the searching algorithm
    input_delta_z = searcher(generator, inverter, classifier, input, y_pred)["delta_z"]
    # If delta_z is smaller than the threshold, we guess that the point is
    # not coming from the known population
    if input_delta_z < threshold:
        return 0
    else:
        return 1

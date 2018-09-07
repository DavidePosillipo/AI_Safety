class NotReliableClassification(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# VAE approach
def conditional_classifier_VAE(input, threshold, VAE, loss_function, classifier):
    # Getting the prediction of the blak box classifier for the input
    y_pred = classifier(input)
    # Getting the reconstructed input and the parameters of the VAE
    recon, mu, logvar = VAE(input)
    # Computing the VAE loss for the input
    input_loss = loss_function(recon, input, mu, logvar)
    # If the input loss is bigger than the threshold, we don't perform any classification
    if input_loss > threshold:
        raise NotReliableClassification("Not reliable classification", y_pred)
    else:
        return y_pred

# WGAN+Inverter approach
def conditional_classifier_WGAN(input, threshold, generator, inverter, classifier, searcher):
    # Getting the prediction of the blak box classifier for the input
    y_pred = classifier(input)
    # Getting the delta_z for the input, using generator, inverter and the searching algorithm
    input_delta_z = searcher(generator, inverter, classifier, input, y_pred)["delta_z"]
    # If delta_z is smaller than the threshold, we don't perform any classification
    if input_delta_z < threshold:
        raise NotReliableClassification("Not reliable classification", y_pred)
    else:
        return y_pred

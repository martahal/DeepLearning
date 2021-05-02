#from Project4.utils import to_cuda
import utils

import pathlib
import torch
import collections

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
#from pyro.optim import adam

class Trainer:

    def __init__(self,
                 batch_size,
                 lr,
                 epochs,
                 model,
                 data,
                 loss_function,
                 optimizer,
                 early_stop_count,
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model = model
        #Transfer model to GPU VRAM if possible
        self.model = utils.to_cuda(self.model)
        (self.training_data, self.test_data) = data

        self.validation_at_step = len(self.training_data)//2 # Validate model after every half epoch.

        # Decide loss function
        if loss_function == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss_function == 'binary_cross_entropy':
            self.loss_function = torch.nn.BCELoss()
        elif loss_function == 'MSE':
            self.loss_function = torch.nn.MSELoss()
        else:
            # TODO
            raise NotImplementedError('This loss function is not implemented yet')

        # Print model to command line
        print(self.model)

        # Define optimizer
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             self.lr)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              self.lr)
        else:
            raise NotImplementedError('Optimizer not implemented')

        self.checkpoint_dir = pathlib.Path("checkpoints")
        self.early_stop_count = early_stop_count


        # Tracking variables
        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()

        )
        self.validation_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()
        )
        self.global_step = 0


    def do_autoencoder_train(self):
        print('\nGENERATIVE AUTOENCODER TRAINING\n')
        for epoch in range(self.epochs):
            # must unpack both images and labels, but we do nothing with the labels
            for (images, classes) in self.training_data:
                train_loss = self._autoencoder_training_step(images, classes)
                self.train_history['loss'][self.global_step] = train_loss
                self.global_step += 1
                if self.global_step % self.validation_at_step == 0:
                    val_loss, accuracy = self._validation(epoch, is_autoencoder=True)
                    self.validation_history['loss'][self.global_step] = val_loss
                    self.save_model() # Saving model
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def do_VAE_train(self):
        print('\nVAE TRAINING\n')
        for epoch in range(self.epochs):
            for (images, classes) in self.training_data:
                elbo_loss = self._VAE_training_step(images)



    def _VAE_training_step(self, images):
        # Transfer to GPU if available
        images = utils.to_cuda(images)
        # Doing the forward pass outside of the forward method in the VAE model
        encoded_x = self.model.encoder(images)

        # Parametrize Q(z|x)
        mu, log_variance = self.model.mu_layer(encoded_x), self.model.variance_layer(encoded_x)  # TODO Why is this called log_var?
        sigma = torch.exp(log_variance / 2)
        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample()

        # Make reconstructions through decoder
        x_hat = self.model.decoder(z)

        # Get Gaussian likelihood reconstruction loss
        reconstruction_loss = self._calculate_reconstruction_loss(x_hat, self.model.log_scale, images)

        # Get KL Divergence
        kl_div = self._calculate_KL_divergence(z, mu, sigma)

        # Provide ELBO loss
        elbo = kl_div - reconstruction_loss
        elbo = elbo.mean() # TODO why?

        # TODO what about the following?
        # Backward pass
        #elbo.backward()
        self.optimizer.step()

        # Reset gradients
        self.optimizer.zero_grad()

        return elbo

    def _autoencoder_training_step(self, images, classes):
        """
        Performs the forward and backward pass of the incoming batch of images through the autoencoder
        :param images: Batch of images
        :return: Training loss
        """
        #Transfer data to GPU VRAM if possible
        images = utils.to_cuda(images)
        # Forward pass through autoencoder
        reconstructed_images, aux = self.model(images)

        # Calculating loss
        # UNCOMMENT TO VISUALIZE INPUT TO CHECK CORRECTNESS
        #show_images_and_reconstructions(images.detach().numpy(), reconstructed_images.detach().numpy(), classes.detach().numpy())
        #plt.show()
        train_loss = self.loss_function(reconstructed_images, images) # images are the targets in this case

        # Backward pass
        train_loss.backward()
        self.optimizer.step()

        # Reset gradients
        self.optimizer.zero_grad()

        return train_loss


    def _validation(self, epoch, is_autoencoder:bool):
        # Set module in evaluation mode
        self.model.eval()
        if is_autoencoder:
            #loss = self._calculate_autoencoder_loss(self.d2_val_dataloader, self.model, self.loss_function)
            loss = self._calculate_autoencoder_loss(self.test_data, self.model, self.loss_function)
            accuracy = 'N/A'
        else:
            loss, accuracy = self._calculate_loss_and_accuracy(self.d2_val_dataloader, self.model, self.loss_function)

        print(f'Epoch: {epoch:>1}',
              f'Iteration: {self.global_step}',
              f'Validation loss: {loss}',
              f'Validation accuracy {accuracy}',
              sep=', ')
        # Set model back into training mode
        self.model.train()

        return loss, accuracy



    def _calculate_autoencoder_loss(self,
                                    data,
                                    #dataloader: torch.utils.data.DataLoader,
                                    model: torch.nn.Module,
                                    loss_criterion: torch.nn.modules.loss._Loss
                                    ):
        average_loss = 0
        with torch.no_grad():
            # must unpack both images and labels, but we do nothing with the labels
            for (images, classes) in self.test_data:
                # Transfer data to GPU VRAM if possible
                images = utils.to_cuda(images)
                reconstructed_images, aux = model(images)

                # Calculate loss
                average_loss += loss_criterion(reconstructed_images, images) # images are the targets in this case
            average_loss /= len(self.test_data)

        return round(average_loss.item(), 4)

    def _calculate_reconstruction_loss(self, x_hat, log_scale, images):
        # Calculate the Gaussian likelihood
        scale = torch.exp(log_scale)
        mean = x_hat
        distribution = torch.distributions.Normal(mean, scale)

        # probability of image under p(x|z)
        log_pxz = distribution.log_prob(images)
        return log_pxz

    def _calculate_KL_divergence(self, z, mu, sigma):
        # Assuming normal distributions:
        # Make fixed normal distribution
        zeros = torch.zeros_like(mu)
        ones = torch.zeros_like(sigma)
        p = torch.distributions.Normal(zeros, ones)

        # Make the estimated distribution from our parameters
        q = torch.distributions.Normal(mu, sigma)

        # Get log probabilities
        log_p, log_q = p.log_prob(z), q.log_prob(z)

        # Calculating kl_div
        kl_div = (log_q - log_p)
        # Summation trick
        #TODO Understand what this does
        kl_div = kl_div.sum(-1)
        return kl_div
    """
    Methods for saving and loading model
    """
    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            val_loss = self.validation_history["loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        return self.model.load_state_dict(state_dict)

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False




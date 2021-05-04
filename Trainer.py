import utils

import pathlib
import torch
from torch import nn
import collections

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
                 is_vae = False,
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
        elif loss_function == 'elbo':
            self.loss_function = None # This is handled elsewhere
        else:
            # TODO
            raise NotImplementedError('This loss function is not implemented yet')

        # Define optimizer
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             self.lr)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              self.lr)
        else:
            raise NotImplementedError('Optimizer not implemented')

        # Print model to command line
        print(self.model)


        self.checkpoint_dir = pathlib.Path("checkpoints")
        self.early_stop_count = early_stop_count


        # Tracking variables
        self.train_history = dict(
            loss=collections.OrderedDict(),
            kl_div=collections.OrderedDict(),
            reconstruction_loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()

        )
        self.validation_history = dict(
            loss=collections.OrderedDict(),
            kl_div=collections.OrderedDict(),
            reconstruction_loss=collections.OrderedDict(),
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
            epoch_loss = 0.
            for (images, classes) in self.training_data:
                #images = utils.to_cuda(images)
                #x_hat, mean, log_std = self.model(images)
                train_loss, kl_div, recon_loss = self._VAE_training_step(images)#x_hat, mean, log_std)

                self.train_history['loss'][self.global_step] = train_loss
                self.train_history['kl_div'][self.global_step] = kl_div
                self.train_history['reconstruction_loss'][self.global_step] = recon_loss
                self.global_step += 1
                if self.global_step % self.validation_at_step == 0:
                    val_loss, val_kl_div, val_recon_loss  = self._validation(epoch, is_autoencoder=False)
                    self.validation_history['loss'][self.global_step] = val_loss
                    self.validation_history['kl_div'][self.global_step] = val_kl_div
                    self.validation_history['reconstruction_loss'][self.global_step] = val_recon_loss
                    self.save_model()  # Saving model
                    if self.should_early_stop():
                        print("Early stopping.")
                        return




    def _VAE_training_step(self, images): # , mean, log_std):
        # Transfer to GPU if available
        images = utils.to_cuda(images)

        x_hat, mean, log_std, encoded_x = self.model(images)

        #reconstruction_loss = self._calculate_reconstruction_loss(x_hat, images)

        kl_div = self._calculate_KL_divergence(mean, log_std, encoded_x)

        reconstruction_loss = self._calculate_reconstruction_loss(x_hat, images, log_scale=nn.Parameter(torch.Tensor([0.0])))

        #elbo = -1 * (kl_div + reconstruction_loss).mean() / images.shape[0]
        #elbo = -1 * (kl_div + reconstruction_loss)

        elbo = kl_div - reconstruction_loss
        elbo = elbo.mean()
        #total_elbo_loss = elbo/images.shape[0]

        # For tracking kl_div and reconstruction loss individually:
        kl_div = kl_div.mean()
        #total_kl_div = kl_div/images.shape[0]

        reconstruction_loss = reconstruction_loss.mean()
        #total_recon_loss = reconstruction_loss/images.shape[0]
        ## Parametrize Q(z|x)
        #mu, log_variance = self.model.mu_layer(encoded_x), self.model.variance_layer(encoded_x)  # TODO Why is this called log_var?
        #sigma = torch.exp(log_variance / 2)
        #q = torch.distributions.Normal(mu, sigma)
        #z = q.rsample()
#
        ## Make reconstructions through decoder
        #x_hat = self.model.decoder(z)
#
        ## Get Gaussian likelihood reconstruction loss
        #reconstruction_loss = self._calculate_reconstruction_loss(x_hat, self.model.log_scale, images)
#
        ## Get KL Divergence
        #kl_div = self._calculate_KL_divergence(mean, log_std) # z, mu, sigma)



        ## Provide ELBO loss
        #elbo = kl_div - reconstruction_loss
        #elbo = elbo.mean() # TODO why?
#
        ## TODO what about the following?
        ## Backward pass
        elbo.backward()
        #total_elbo_loss.backward()
        self.optimizer.step()

        # Reset gradients
        self.optimizer.zero_grad()
        return elbo, kl_div, reconstruction_loss
        #return total_elbo_loss, total_kl_div, total_recon_loss

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
            loss = self._calculate_autoencoder_loss(self.test_data, self.model, self.loss_function)
            accuracy = 'N/A'
            print(f'Epoch: {epoch:>1}',
                  f'Iteration: {self.global_step}',
                  f'Validation loss: {loss}',
                  f'Validation accuracy {accuracy}',
                  sep=', ')
            self.model.train()
            return loss, accuracy
        else:
            loss, kl_div, recon_loss = self._calculate_vae_loss()
            accuracy = 'N/A'
            print(f'Epoch: {epoch:>1}',
                  f'Iteration: {self.global_step}',
                  f'Validation loss: {loss}',
                  f'Validation KL_div {kl_div}',
                  f'Validation reconstruction loss {recon_loss}',
                  sep=', ')
            # Set model back into training mode
            self.model.train()
            return loss, kl_div, recon_loss

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

    def _calculate_vae_loss(self):
        with torch.no_grad():
            #elbo_test_loss = 0.
            elbo = 0.
            total_kl_div = 0.
            total_recon_loss = 0.
            for (images, classes) in self.test_data:
                images = utils.to_cuda(images)
                #x_hat, mean, log_std = self.model(images)
                x_hat, mean, log_std, encoded_x = self.model(images)


                #reconstruction_loss = self._calculate_reconstruction_loss(x_hat, images, nn.)
                reconstruction_loss = self._calculate_reconstruction_loss(x_hat, images,
                                                                          log_scale=nn.Parameter(torch.Tensor([0.0])))

                #kl_div = self._calculate_KL_divergence(mean, log_std, z)
                kl_div = self._calculate_KL_divergence(mean, log_std, encoded_x)

                #elbo_test_loss += -1 * (kl_div + reconstruction_loss).mean() / images.shape[0]
                elbo += (kl_div - reconstruction_loss).mean()#/images.shape[0]

                # for tracking kl_div and recon loss:
                total_kl_div += kl_div.mean()
                total_recon_loss += reconstruction_loss.mean()

            total_kl_div = total_kl_div / len(self.test_data)
            total_recon_loss = total_recon_loss / len(self.test_data)

            total_test_loss = elbo/len(self.test_data)
        return round(total_test_loss.item(), 4), round(total_kl_div.item(), 4), round(total_recon_loss.item(), 4)
        #total_test_loss = elbo_test_loss / len(self.test_data)
        #return round(total_test_loss.item(), 4)


    def _calculate_reconstruction_loss(self, x_hat, images, log_scale=None):
        if log_scale is not None:
            # Calculate the Gaussian likelihood
            log_scale = utils.to_cuda(log_scale)
            scale = torch.exp(log_scale)
            mean = x_hat
            p_xz = torch.distributions.Normal(mean, scale)
            # probability of image under p(x|z)
            log_pxz = p_xz.log_prob(images)
            log_pxz = log_pxz.sum(dim=(1, 2, 3))
            return log_pxz
        else:
            #Calculate log likelihood assuming multivariate Bernoulli distribution
            recon_loss = torch.sum((images * torch.log(x_hat + 1e-9)) + \
            (1 - images) * torch.log(1 - x_hat + 1e-9), axis=(1, 2, 3))
            return recon_loss

    def _calculate_KL_divergence(self, mean, log_std, z):#z, mu, sigma):

        #kl_div = 0.5 * torch.sum(1 + torch.log(log_std.exp() ** 2 + 1e-9) - mean.pow(2) - log_std.exp() ** 2, axis=1)
        #return kl_div
        # Assuming normal distributions:
        # Make fixed normal distribution
        zeros = torch.zeros_like(mean)
        ones = torch.ones_like(log_std)
        p = torch.distributions.Normal(zeros, ones)
#
        ## Make the estimated distribution from our parameters
        q = torch.distributions.Normal(mean, log_std)
#
        ## Get log probabilities
        log_p, log_q = p.log_prob(z), q.log_prob(z)
#
        ## Calculating kl_div
        kl_div = (log_q - log_p)
        ## Summation trick
        ##TODO Understand what this does
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
        self.model.load_state_dict(state_dict)

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





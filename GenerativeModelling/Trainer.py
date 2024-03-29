from GenerativeModelling import utils

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
                 model_save_path,
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


        self.checkpoint_dir = pathlib.Path(model_save_path)
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
                train_loss, kl_div, recon_loss = self._VAE_training_step(images)

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

        # Make reconstructions through decoder
        x_hat, mean, log_std, encoded_x = self.model(images)
        # Calculate kl divergence
        kl_div = self._calculate_KL_divergence(mean, log_std, encoded_x)
        # # Get Gaussian likelihood reconstruction loss
        reconstruction_loss = self._calculate_reconstruction_loss(x_hat, images, log_scale=nn.Parameter(torch.Tensor([0.0])))

        # Provide ELBO loss
        elbo = kl_div - reconstruction_loss
        elbo = elbo.mean()

        # For tracking kl_div and reconstruction loss individually:
        kl_div = kl_div.mean()
        reconstruction_loss = reconstruction_loss.mean()

        ## Backward pass
        elbo.backward()
        self.optimizer.step()

        # Reset gradients
        self.optimizer.zero_grad()
        return elbo, kl_div, reconstruction_loss


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

    def ae_detect_anomaly_by_loss(self):
        # validate all images in test data set
        # make a list of reconstruction losses
        self.model.eval()
        images = None
        recons= None
        losses = None

        with torch.no_grad():
            for image_batch, label_batch in self.test_data:
                #image = self.test_data[i][0][j]  # each batch is a tuple of images and labels
                # make reconstruction
                # transfer to GPU if possible:
                image_batch = utils.to_cuda(image_batch)
                reconstruction_batch, aux = self.model(image_batch)
                recon_loss = ((image_batch - reconstruction_batch) ** 2).sum(axis=(1, 2, 3))/ (28 * 28) # Divide by image size
                #transform images and reconstructions to plot them
                image_batch = image_batch.view(
                    image_batch.shape[0],
                    image_batch.shape[2],
                    image_batch.shape[3],
                    image_batch.shape[1]
                )
                reconstruction_batch = reconstruction_batch.view(
                    reconstruction_batch.shape[0],
                    reconstruction_batch.shape[2],
                    reconstruction_batch.shape[3],
                    reconstruction_batch.shape[1]
                )


                images = torch.cat([images, image_batch]) if images is not None else image_batch# TODO determine wether to detach or not
                recons = torch.cat([recons, reconstruction_batch]) if recons is not None else reconstruction_batch
                losses = torch.cat([losses, recon_loss]) if losses is not None else recon_loss
        self.model.train()
        return images.cpu().detach().numpy(), recons.cpu().detach().numpy(), losses.cpu().detach().numpy()

    def vae_detect_anomaly_by_loss(self):
        self.model.eval()
        # validate all images in test data set
        # make a list of reconstruction losses
        images = None
        recons= None
        losses = None

        with torch.no_grad():
            for image_batch, label_batch in self.test_data:
                # make reconstruction
                # transfer to GPU if possible:
                image_batch = utils.to_cuda(image_batch)
                reconstruction_batch, mean, log_std, encoded_x = self.model(image_batch)
                recon_loss = ((image_batch - reconstruction_batch) ** 2).sum(axis=(1, 2, 3))/ (28 * 28) # Divide by image size
                # transform images and reconstructions to plot them
                image_batch = image_batch.view(
                    image_batch.shape[0],
                    image_batch.shape[2],
                    image_batch.shape[3],
                    image_batch.shape[1]
                )
                reconstruction_batch = reconstruction_batch.view(
                    reconstruction_batch.shape[0],
                    reconstruction_batch.shape[2],
                    reconstruction_batch.shape[3],
                    reconstruction_batch.shape[1]
                )
                images = torch.cat([images, image_batch]) if images is not None else image_batch
                recons = torch.cat([recons, reconstruction_batch]) if recons is not None else reconstruction_batch
                losses = torch.cat([losses, recon_loss]) if losses is not None else recon_loss
        self.model.train()
        return images.cpu().detach().numpy(), recons.cpu().detach().numpy(), losses.cpu().detach().numpy()


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

    def _calculate_KL_divergence(self, mean, log_std, z):

        # Assuming normal distributions:
        # Make fixed normal distribution
        zeros = torch.zeros_like(mean)
        ones = torch.ones_like(log_std)
        p = torch.distributions.Normal(zeros, ones)

        # Parametrize q(z|x)
        ## Make the estimated distribution from our parameters
        q = torch.distributions.Normal(mean, log_std)
#
        ## Get log probabilities from p(z) and q(z|x)
        log_p, log_q = p.log_prob(z), q.log_prob(z)
#
        ## Calculating kl_div
        kl_div = (log_q - log_p)
        ## Summation trick
        kl_div = kl_div.sum(-1)
        return kl_div





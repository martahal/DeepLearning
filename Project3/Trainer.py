import torch
import collections
import numpy as np

class Trainer:

    def __init__(self,
                 batch_size,
                 lr,
                 epochs,
                 model,
                 dataloaders,
                 loss_function,
                 optimizer,
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.lr = lr
        self. epochs = epochs
        self.model = model
        self.dataloaders = dataloaders

        # Decide loss function
        if loss_function == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss_function == 'binary_cross_entropy':
            self.loss_function = torch.nn.BCELoss()
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

        # Load dataset:
        self.d1_train_dataloader, \
        self.d2_train_dataloader, \
        self.d2_val_dataloader, \
        self.d2_test_dataloader  \
            = dataloaders

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

    def do_classifier_train(self, freeze_encoder_weights=False):
        print('\nCLASSIFIER TRAINING\n')
        for epoch in range(self.epochs):
            for X_batch, Y_batch in self.d2_train_dataloader:
                train_loss, train_accuracy = self._classifier_training_step(X_batch, Y_batch)
                self.train_history['loss'][self.global_step] = train_loss
                self.train_history['accuracy'][self.global_step] = train_accuracy
                self.global_step += 1

            # Validate model for each epoch
            val_loss, val_accuracy = self._validation(epoch, is_autoencoder=False)
            self.validation_history['loss'][self.global_step] = val_loss
            self.validation_history['accuracy'][self.global_step] = val_accuracy
        # Test model after all epochs
        self.test_loss, self.test_accuracy = self._calculate_loss_and_accuracy(self.d2_test_dataloader,
                                                                               self.model,
                                                                               self.loss_function)
        print(f'Test loss: {self.test_loss}',
              f'Test accuracy: {self.test_accuracy}',
              sep= ', ')

    def do_autoencoder_train(self):
        print('\nAUTOENCODER TRAINING\n')
        for epoch in range(self.epochs):
            # must unpack both images and labels, but we do nothing with the labels
            for images, labels in self.d1_train_dataloader: # dataset should be D1

                train_loss = self._autoencoder_training_step(images)
                self.train_history['loss'][self.global_step] = train_loss
                self.global_step += 1
            # Validate model after 600 iterations
                if self.global_step % 600 == 0:
                    # accuracy is not used
                    val_loss, accuracy = self._validation(epoch, is_autoencoder=True)
                    self.validation_history['loss'][self.global_step] = val_loss
        # TODO Resolve whether to test or not

    def _classifier_training_step(self, X_batch, Y_batch):
        """
        Performs forward and backward pass for each incoming batch.
        :param X_batch: Batch of images
        :param Y_batch: Batch of labels
        :return: Training loss: value of training loss. Training accuracy: fraction of batch that was correctly labelled
        """
        # Forward pass through model
        output_probs, aux = self.model(X_batch) # produces a tuple for some reason. fix found here: https://github.com/pytorch/vision/issues/302#issuecomment-341163548

        # Calculating loss
        train_loss = self.loss_function(output_probs, Y_batch)

        # Calculating accuracy
        predictions = torch.argmax(output_probs, dim=1)
        total_correct = (predictions == Y_batch).sum().item()
        total_images = predictions.shape[0]
        train_accuracy = total_correct / total_images

        # Backward pass
        train_loss.backward()
        self.optimizer.step()

        # Reset gradients
        self.optimizer.zero_grad()

        return train_loss, train_accuracy

    def _autoencoder_training_step(self, images):
        """
        Performs the forward and backward pass of the incoming batch of images through the autoencoder
        :param images: Batch of images
        :return: Training loss
        """
        # Forward pass through autoencoder
        reconstructed_images, aux = self.model(images)

        # Calculating loss
        train_loss = self.loss_function(reconstructed_images, images)

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
            loss = self._calculate_autoencoder_loss(self.d2_val_dataloader, self.model, self.loss_function)
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

    def _calculate_loss_and_accuracy(self,
                                     dataloader: torch.utils.data.DataLoader,
                                     model: torch.nn.Module,
                                     loss_criterion: torch.nn.modules.loss._Loss
                                     ):
        average_loss = 0
        total_correct = 0
        total_images = 0
        with torch.no_grad():
            for (X_batch, Y_batch) in dataloader:
                # Forward pass the images through our model
                output_probs, aux = model(X_batch) # produces a tuple for some reason. fix found here: https://github.com/pytorch/vision/issues/302#issuecomment-341163548

                # Calculate Loss
                average_loss += loss_criterion(output_probs, Y_batch)
                # Calculate accuracy
                predictions = torch.argmax(output_probs, dim=1)
                total_correct += (predictions == Y_batch).sum().item()
                total_images += predictions.shape[0]

            average_loss /= len(dataloader)
            accuracy = total_correct / total_images

        return round(average_loss.item(), 4), round(accuracy, 4)

    def _calculate_autoencoder_loss(self,
                                    dataloader: torch.utils.data.DataLoader,
                                    model: torch.nn.Module,
                                    loss_criterion: torch.nn.modules.loss._Loss
                                    ):
        average_loss = 0
        with torch.no_grad():
            # must unpack both images and labels, but we do nothing with the labels
            for images, labels in dataloader:
                # Forward pass images through the autoencoder to retrieve the reconstructed images
                reconstructed_images, aux = model(images)

                # Calculate loss
                average_loss += loss_criterion(reconstructed_images, images)
            average_loss /= len(dataloader)

        return round(average_loss.item(), 4)


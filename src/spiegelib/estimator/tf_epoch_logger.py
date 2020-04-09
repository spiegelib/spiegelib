#!/usr/bin/env python
"""
Logger class for logging accuracy and loss from epochs when training a
``tf.keras.Model``. Inherits from ``tf.keras.callbacks.Callback`` and should
be passed in as a callback during model training.
"""

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class TFEpochLogger(Callback):
    """
    Attributes:
        log_data (dict): Dictionary of data logged at the end of each epoch
            during training. Keyed on the epoch number.
    """

    def __init__(self):
        """
        Constructor
        """

        self.log_data = {}
        super().__init__()


    def on_epoch_end(self, epoch, logs=None):
        """
        Logging on end of an epoch. This is called automatically during training.
        Stores infomation passed in the logs parameter into ``log_data`` attribute.

        Args:
            epoch (int): current epoch that just finished
            logs: Any data to be logged
        """

        self.log_data[epoch] = logs


    def get_plotting_data(self):
        """
        Helper function to organize log_data for easier plotting
        """

        plot_data = {}
        epochs = []
        for epoch in self.log_data:
            epochs.append(epoch + 1)
            for key in self.log_data[epoch]:
                if key in plot_data:
                    plot_data[key].append(self.log_data[epoch][key])
                else:
                    plot_data[key] = [self.log_data[epoch][key]]

        return epochs, plot_data


    def plot(self):
        """
        Plot logged training and validation data using matplotlib
        """

        epochs, plot_data = self.get_plotting_data()

        train_accuracy = 'accuracy' in plot_data
        train_loss = 'loss' in plot_data
        val_accuracy = 'val_accuracy' in plot_data
        val_loss = 'val_loss' in plot_data

        if train_accuracy and train_loss:
            fig, axs = plt.subplots(2)
            fig.suptitle('Model Accuracy and Loss')

            axs[0].plot(epochs, plot_data['accuracy'], label="Train Accuracy")
            if val_accuracy:
                axs[0].plot(epochs, plot_data['val_accuracy'],
                            label="Validation Accuracy")

            axs[0].set(ylabel='Accuracy (%)')
            axs[0].legend()

            axs[1].plot(epochs, plot_data['loss'], label="Train Loss")
            if val_loss:
                axs[1].plot(epochs, plot_data['val_loss'], label="Validation Loss")

            axs[1].set(xlabel='Epochs', ylabel='Loss')
            axs[1].legend()

            plt.show()

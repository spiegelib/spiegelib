#!/usr/bin/env python
"""
Logger class for logging accuracy and loss from epochs when training a
TensoFlow Model
"""

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class TFEpochLogger(Callback):

    def __init__(self):
        """
        Constructor
        """

        self.logData = {}
        super().__init__()


    def on_epoch_end(self, epoch, logs=None):
        """
        Logging on end of an epoch
        """

        self.logData[epoch] = logs


    def get_plotting_data(self):
        """
        Helper function to organize logData for easier plotting
        """

        plotData = {}
        epochs = []
        for epoch in self.logData:
            epochs.append(epoch + 1)
            for key in self.logData[epoch]:
                if key in plotData:
                    plotData[key].append(self.logData[epoch][key])
                else:
                    plotData[key] = [self.logData[epoch][key]]

        return epochs, plotData


    def plot(self):
        """
        Plot logged training and validation data using matplotlib
        """

        epochs, plotData = self.get_plotting_data()

        plotTrainAccuracy = 'accuracy' in plotData
        plotTrainLoss = 'loss' in plotData
        plotValAccuracy = 'val_accuracy' in plotData
        plotValLoss = 'val_loss' in plotData

        if plotTrainAccuracy and plotTrainLoss:
            fig, axs = plt.subplots(2)
            fig.suptitle('Model Accuracy and Loss')

            axs[0].plot(epochs, plotData['accuracy'], label="Train Accuracy")
            if plotValAccuracy:
                axs[0].plot(epochs, plotData['val_accuracy'], label="Validation Accuracy")

            axs[0].set(ylabel='Accuracy (%)')
            axs[0].legend()

            axs[1].plot(epochs, plotData['loss'], label="Train Loss")
            if plotValLoss:
                axs[1].plot(epochs, plotData['val_loss'], label="Validation Loss")

            axs[1].set(xlabel='Epochs', ylabel='Loss')
            axs[1].legend()

            plt.show()

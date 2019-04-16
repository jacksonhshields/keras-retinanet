import keras.callbacks
import neptune

from keras.callbacks import Callback

class NeptuneMonitor(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if 'loss' in logs:
            neptune.send_metric('loss', epoch, logs['loss'])
        if 'classification_loss' in logs:
            neptune.send_metric('classification_loss', epoch, logs['classification_loss'])
        if 'regression_loss' in logs:
            neptune.send_metric('regression_loss', epoch, logs['regression_loss'])
        if 'mAP' in logs:
            neptune.send_metric('mAP', epoch, logs['mAP'])

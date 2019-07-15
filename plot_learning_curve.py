import tensorflow as tf
class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.figure(figsize=(25,10))
        plt.plot(self.x, self.losses, label="loss", marker='o')
        plt.plot(self.x, self.val_losses, label="val_loss", marker='o')
        plt.legend()
        plt.xticks([i for i in range(epochs)])
        plt.show()

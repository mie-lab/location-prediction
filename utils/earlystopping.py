import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logdir, patience=7, verbose=False, main_process=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.log_dir = logdir
        self.patience = patience
        self.verbose = verbose
        self.main_process = main_process
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_return_dict = {"val_loss": np.inf}
        self.delta = delta

    def __call__(self, return_dict, model):

        score = return_dict["val_loss"]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            return

        if score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose and self.main_process:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, return_dict, model):
        """Saves model when validation loss decrease."""
        if self.verbose and self.main_process:
            print(
                f"Validation loss decreased ({self.best_return_dict['val_loss']:.6f} --> {return_dict['val_loss']:.6f}).  Saving model ..."
            )
        if self.main_process:
            torch.save(model.state_dict(), self.log_dir + "/checkpoint.pt")
        self.best_return_dict = return_dict

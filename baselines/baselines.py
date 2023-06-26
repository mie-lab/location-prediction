import torch
import numpy as np


class baselines:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.persistent_forecast()
        self.frequent_forecast()
        self.markov_forecast_local()

    def persistent_forecast(self):
        """Predict the same value as the previous timestep (data==target, lower bound)."""
        correct = 0
        total = 0

        for _, (data, target, _) in enumerate(self.train_loader):
            correct += (data[-1] == target).sum().numpy()
            total += 1

        print("Persistent forecast train accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (data, target, _) in enumerate(self.val_loader):
            correct += (data[-1] == target).sum().numpy()
            total += 1

        print("Persistent forecast validation accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (data, target, _) in enumerate(self.test_loader):
            correct += (data[-1] == target).sum().numpy()
            total += 1

        print("Persistent forecast test accuracy = {:.2f}".format(100 * correct / total))

    def frequent_forecast(self):
        """Predict the most frequent value as the target."""
        correct = 0
        total = 0
        for _, (data, target, _) in enumerate(self.train_loader):
            output, counts = torch.unique(data, sorted=False, return_counts=True)
            predict = output[counts.argmax()]

            correct += (predict == target).sum().numpy()
            total += 1

        print("Frequent forecast train accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (data, target, _) in enumerate(self.val_loader):
            output, counts = torch.unique(data, sorted=False, return_counts=True)
            predict = output[counts.argmax()]

            correct += (predict == target).sum().numpy()
            total += 1

        print("Frequent forecast validation Accuracy = {:.2f}".format(100 * correct / total))

        correct = 0
        total = 0
        for _, (data, target, _) in enumerate(self.test_loader):
            output, counts = torch.unique(data, sorted=False, return_counts=True)
            predict = output[counts.argmax()]

            correct += (predict == target).sum().numpy()
            total += 1

        print("Frequent forecast test Accuracy = {:.2f}".format(100 * correct / total))

    def markov_forecast_self(self, loader):
        correct = 0
        total = 0
        for _, (data, target, _) in enumerate(loader):
            # most frequent
            # output, counts = torch.unique(data, sorted=False, return_counts=True)
            # most_frequent = output[counts.argmax()]

            # markov matrix construction
            idx, inverse_indices = data.unique(return_inverse=True)
            mark_matrix = np.zeros([idx.shape[0], idx.shape[0]])
            for i in range(data.shape[0] - 1):
                mark_matrix[inverse_indices[i], inverse_indices[i + 1]] += 1

            current_predict = np.argmax(mark_matrix[inverse_indices[-1], :])
            if current_predict != 0:
                current_predict = idx[current_predict]

            correct += (current_predict == target).numpy()[0]
            total += 1

        print("Accuracy = {:.2f}".format(100 * correct / total))

    def markov_forecast_local(self):
        """Predict according to the previous experience, otherwise predict the most frequent value as the target."""
        self.markov_forecast_self(self.train_loader)
        self.markov_forecast_self(self.val_loader)
        self.markov_forecast_self(self.test_loader)

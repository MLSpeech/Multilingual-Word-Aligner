import numpy as np
import torch

class LossFunction_DP:
    def __init__(self, w, C, features_object, train_cfg):
        self.w = w
        self.C = C
        self.features_object = features_object
        self.train_cfg = train_cfg
        self.update = np.zeros(len(features_object.functions_to_run))
        self.num_updates = 0
        self.num_iteration_with_no_change = 0

    def gamma(self, y, y_pred):
        # Define the cost function γ(y, y')
        y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
        y_pred = torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
        return torch.sum(torch.abs(y - y_pred)) / len(y)

    def compute_loss(self, yi, y_pred, a):
        cost = self.gamma(yi, y_pred)
        loss = torch.maximum(torch.sqrt(cost) - torch.dot(self.w, a), torch.tensor(0.0))
        return loss

    def update_weights(self, loss, a):
        if loss > 0:
            step_size = min(loss / np.dot(a, a), self.C)
            self.w += step_size * a
            self.num_iteration_with_no_change = 0
        else:
            self.num_iteration_with_no_change += 1



def calc_L1LossFunction(y, y_pred):
    # Ensure y and y_pred are PyTorch tensors
    y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
    y_pred = torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    
    # Calculate the L1 loss (Mean Absolute Error)
    return torch.sum(torch.abs(y - y_pred)) / y.size(0)
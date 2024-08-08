import numpy as np
import torch
import copy

def model_to_vec(model):
    vec = np.array([])
    with torch.no_grad():
        for p in model.parameters():
            vec = np.append(vec, p.data.clone().detach().flatten().numpy())

    return vec


class LearningrateEstimatorModel:
    def __init__(self, base_lr, b1=0.5, b2=0.5, b3=0.5):
        self.initial_lr = base_lr
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        self.model_ema = None
        self.prev_model_ema = None
        self.prev_model_ema_na = None
        self.variance_ema = 0
        self.prev_variance_ema = 0
        self.prev_variance_ema_na = 0
        self.variance_ratio_ema = 0
        self.prev_variance_ratio_ema_na = 0

    def initialize(self, model):
        self.model_ema = np.zeros(len(model_to_vec(model)))
        self.prev_model_ema = copy.deepcopy(self.model_ema)
        self.prev_model_ema_na = copy.deepcopy(self.model_ema)

    def estimate(self, model, current_round, base_lr):
        if self.model_ema is None:
            self.initialize(model)

        model_vec = model_to_vec(model)
        # calculate ema on the loss mean
        self.model_ema = self.b1 * self.prev_model_ema_na + (1 - self.b1) * model_vec
        self.prev_model_ema_na = copy.deepcopy(self.model_ema)
        # bias correction
        self.model_ema = self.model_ema / (1 - pow(self.b1, current_round))


        # ema on the loss variance
        self.variance_ema = (self.b2 * self.prev_variance_ema_na + (1 - self.b2) *
                             np.mean((model_vec - self.prev_model_ema) * (model_vec - self.prev_model_ema)))
        self.prev_variance_ema_na = copy.deepcopy(self.variance_ema)
        # bias correction
        self.variance_ema = self.variance_ema / (1 - pow(self.b2, current_round))

        # counter divide by 0 error
        if self.prev_variance_ema == 0:
            ratio = 1
        else:
            ratio = (self.variance_ema / self.prev_variance_ema)


        # calculate ema on variance ratio
        self.variance_ratio_ema = self.b3 * self.prev_variance_ratio_ema_na + (1 - self.b3) * ratio
        self.prev_variance_ratio_ema_na = copy.deepcopy(self.variance_ratio_ema)
        # bias correction
        self.variance_ratio_ema = self.variance_ratio_ema / (1 - pow(self.b3, current_round))


        self.prev_model_ema = copy.deepcopy(self.model_ema)
        self.prev_variance_ema = copy.deepcopy(self.variance_ema)

        # calculate lr
        lr = min(self.initial_lr, base_lr * self.variance_ratio_ema)

        return lr


class LearningrateEstimatorLoss:
    def __init__(self, initial_lr, b1=0.7, b2=0.3, b3=0.7):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        self.initial_lr = initial_lr
        self.id = None

        self.loss_ema = 0
        self.prev_loss_ema = 0
        self.prev_loss_ema_na = 0
        self.variance_ema = 0
        self.prev_variance_ema = 0
        self.prev_variance_ema_na = 0
        self.variance_ratio_ema = 0
        self.prev_variance_ratio_ema_na = 0

    def estimate(self, loss, current_round, base_lr):
        # calculate ema on the loss mean
        self.loss_ema = self.b1 * self.prev_loss_ema_na + (1 - self.b1) * loss
        self.prev_loss_ema_na = copy.deepcopy(self.loss_ema)
        # bias correction
        self.loss_ema = self.loss_ema / (1 - pow(self.b1, current_round))

        # ema on the loss variance
        self.variance_ema = (self.b2 * self.prev_variance_ema_na + (1 - self.b2) *
                             (loss - self.prev_loss_ema) * (loss - self.prev_loss_ema))
        self.prev_variance_ema_na = copy.deepcopy(self.variance_ema)
        # bias correction
        self.variance_ema = self.variance_ema / (1 - pow(self.b2, current_round))

        # counter divide by 0 error
        if self.prev_variance_ema == 0:
            ratio = 1
        else:
            ratio = (self.variance_ema / self.prev_variance_ema)

        # calculate ema on variance ratio
        self.variance_ratio_ema = self.b3 * self.prev_variance_ratio_ema_na + (1 - self.b3) * ratio
        self.prev_variance_ratio_ema_na = copy.deepcopy(self.variance_ratio_ema)
        # bias correction
        self.variance_ratio_ema = self.variance_ratio_ema / (1 - pow(self.b3, current_round))

        # copy into prev variables
        self.prev_variance_ema = copy.deepcopy(self.variance_ema)
        self.prev_loss_ema = copy.deepcopy(self.loss_ema)

        # calculate lr
        lr = min(self.initial_lr, base_lr * self.variance_ratio_ema)

        return lr
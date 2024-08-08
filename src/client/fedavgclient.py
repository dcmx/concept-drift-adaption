import copy
import torch
import inspect
import numpy as np
from .baseclient import BaseClient
from src import MetricManager
from src.algorithm.learningrate_estimator import LearningrateEstimatorLoss


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        self.estimator = LearningrateEstimatorLoss(initial_lr=self.args.lr, b1=self.args.b1, b2=self.args.b2, b3=self.args.b3)
        self.round = 0

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)

        if self.id == 97:
            a = 1
        # drift adaptation
        #if self.estimator.model_ema is None:
        #    self.estimator.initialize(self.model)
        #self.args.lr = self.estimator.estimate(self.model, self.round)

        self.train = self.model.train()
        self.model.to(self.args.device)

        #first_round = True
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                #if first_round and self.args.drift_adaptation:
                #    first_round = False
                #    self.args.lr = self.estimator.estimate(loss.item(), self.round, self.args.lr)
                #    for g in optimizer.param_groups:
                #        g['lr'] = self.args.lr

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        return mm.results

    @torch.inference_mode()
    def update_estimator(self):
        self.model.eval()
        self.model.to(self.args.device)

        if self.estimator.id is None:
            self.estimator.id = self.id


        batch_count = 0
        loss_arr = []
        while batch_count <= 50:
            inputs, targets = next(iter(self.train_loader))
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss_arr.append(self.criterion()(outputs, targets).item())
            batch_count += self.args.B

        loss = np.mean(loss_arr)

        self.args.lr = self.estimator.estimate(loss, self.round, self.args.lr)

        return self.args.lr

    @torch.inference_mode()
    def evaluate(self):
        if self.args._train_only: # `args.test_fraction` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):
        self.model = copy.deepcopy(model)

    def upload(self):
        self.model.to('cpu')
        return self.model.named_parameters()
    
    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'

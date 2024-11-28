import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted


import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted


class Optimizer:
    def __init__(self):
        pass

    def start_epoch(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def get_epoch_len(self):
        raise NotImplementedError


class SubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01):
        super(SubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )

    def start_epoch(self):
        pass

    def step(self):
        g = self.objective.get_batch_subgrad(self.weights)
        self.weights = self.weights - self.lr * g

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return 1


class StochasticSubgradientMethodDP(Optimizer):
    def __init__(self, objective, lr=0.01, batch_size=64, seed=25, epoch_len=None,noise=0):
        super(StochasticSubgradientMethodDP, self).__init__()
        self.objective = objective
        self.lr = lr
        self.batch_size = batch_size

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
        self.order = None
        self.iter = None
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size
        self.noise = noise

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0

    def step(self):
        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        # self.weights.requires_grad = True
        g,r_g,coeff = self.objective.get_batch_subgrad_dp(self.weights, idx=idx)

        
        #Adding Noise
        # print("self.noise:",self.noise)
        # print("coeff",coeff)
        g += (torch.normal(mean=0.0, std=self.noise*coeff, size=g.size()))
        #Gradient of Regularizer
        g += r_g 
        # self.weights.requires_grad = False
        self.weights = self.weights - self.lr * g
        # print(self.weights.shape)
        self.iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class StochasticRegularizedDualAveraging(Optimizer):
    def __init__(
        self, objective, lr=0.01, l2_reg=1.0, batch_size=64, seed=25, epoch_len=None
    ):
        super(StochasticRegularizedDualAveraging, self).__init__()
        self.objective = objective
        self.aux_reg = 1 / lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.dual_avg = torch.zeros(
                objective.n_class * self.objective.d, dtype=torch.float64
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
            self.dual_avg = torch.zeros(self.objective.d, dtype=torch.float64)

        self.order = None
        self.epoch_iter = None
        self.total_iter = 0
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.epoch_iter = 0

    def step(self):
        idx = self.order[
            self.epoch_iter
            * self.batch_size : min(
                self.objective.n, (self.epoch_iter + 1) * self.batch_size
            )
        ]
        g = self.objective.get_batch_subgrad(self.weights, idx=idx, include_reg=False)
        self.dual_avg = (self.total_iter * self.dual_avg + g) / (self.total_iter + 1)
        self.weights = -self.dual_avg / (
            self.l2_reg / self.objective.n + self.aux_reg / (self.total_iter + 1)
        )
        self.weights.requires_grad = True
        self.epoch_iter += 1
        self.total_iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class SmoothedLSVRG(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=1.0,
        smoothing="l2",
        seed=25,
        length_epoch=None,
    ):
        super(SmoothedLSVRG, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        self.lr = lr
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
        self.spectrum = self.objective.sigmas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = length_epoch
        else:
            self.length_epoch = int(nb_passes * n)
        self.nb_checkpoints = 0
        self.step_no = 0

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n

        # start epoch
        if self.step_no % n == 0:
            losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
            sorted_losses, self.argsort = torch.sort(losses, stable=True)
            self.sigmas = get_smooth_weights_sorted(
                sorted_losses, self.spectrum, self.smooth_coef, self.smoothing
            )
            with torch.enable_grad():
                self.subgrad_checkpt = self.objective.get_batch_subgrad(self.weights, include_reg=False)
            self.weights_checkpt = torch.clone(self.weights)
            self.nb_checkpoints += 1

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.sigmas)])
        x = self.objective.X[self.argsort[i]]
        y = self.objective.y[self.argsort[i]]

        # Compute gradient at current iterate.
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y).squeeze()

        if self.uniform:
            direction = n * self.sigmas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights / n

        self.weights.copy_(self.weights - self.lr * direction)
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch

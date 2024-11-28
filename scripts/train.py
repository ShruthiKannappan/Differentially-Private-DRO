"""
Train model for a particular objective and optimizer on evry hyperparameter setting.
"""

import time
import datetime
from joblib import Parallel, delayed
import sys
import argparse
import os
L2_REG = 1.0
SHIFT_COST = 1.0
LRS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3]

# Create parser.
sys.path.append(".")
from src.utils.training import (
    OptimizationError,
    compute_training_curve,
    format_time,
    find_best_optim_cfg,
    FAIL_CODE,
)
from src.utils.io import dict_to_list
from src.utils.hyperparams import HYPERPARAM_LR


from dp_accounting.rdp import rdp_privacy_accountant as rdp
from dp_accounting import dp_event as event
from scipy import optimize as opt

def get_noise_multiplier(target_epsilon, target_delta, sampling, step):
    RDP_ORDERS = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )
    def objective(noise_multiplier):
        accountant = rdp.RdpAccountant(RDP_ORDERS)
        dpevent = event.SelfComposedDpEvent(
            event.PoissonSampledDpEvent(
                sampling, event.GaussianDpEvent(noise_multiplier)
            ),
            int(step),
        )
        accountant.compose(dpevent)
        eps = accountant.get_epsilon(target_delta)
        return eps - target_epsilon

    optimal_noise = opt.brentq(objective, 1e-6, 1000)
    print(optimal_noise)
    return optimal_noise

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "yacht",
        "energy",
        "simulated",
        "concrete",
        "iwildcam",
        "kin8nm",
        "power",
        "acsincome",
        "diabetes",
        "amazon",
    ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
    ],
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=64,
)
parser.add_argument(
    "--epoch_len",
    type=int,
    default=None,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
)
parser.add_argument(
    "--dataset_length",
    type=int,
    default=64,
)
parser.add_argument(
    "--use_hyperparam",
    type=int,
    default=0,
)
parser.add_argument("--epsilon", type=float, default=4.0)
parser.add_argument("--delta", type=float, default=None)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--n_jobs", type=int, default=-2)

args = parser.parse_args()

if args.delta is None:
    args.delta = args.dataset_length**(-1.1)

# Configure for input to trainers.
dataset = args.dataset
if dataset in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
    loss = "squared_error"
    n_class = None
elif dataset == "iwildcam":
    loss = "multinomial_cross_entropy"
    n_class = 60
elif dataset == "amazon":
    loss = "multinomial_cross_entropy"
    n_class = 5
elif dataset == "diabetes":
    loss = "binary_cross_entropy"
    n_class = None

model_cfg = {
    "objective": args.objective,
    "l2_reg": L2_REG,
    "shift_cost": SHIFT_COST,
    "loss": loss,
    "n_class": n_class
}


if args.use_hyperparam:
    lrs = [HYPERPARAM_LR[args.optimizer][args.dataset][args.objective]]
else:
    lrs = LRS
optim_cfg = {
    "optimizer": args.optimizer,
    "lr": lrs,
    "epoch_len": args.epoch_len,
    "shift_cost": SHIFT_COST,
    "batch_size":args.batch_size,
    "dataset_length":args.dataset_length,
    "noise": get_noise_multiplier(
    target_epsilon=args.epsilon,
    target_delta=args.delta,
    sampling=args.batch_size / args.dataset_length,
    step=((args.n_epochs) * (args.dataset_length)) / args.batch_size
    )
}
seeds = [1, 2]
n_epochs = args.n_epochs
parallel = bool(args.parallel)

optim_cfgs = dict_to_list(optim_cfg)

config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optim_cfg": optim_cfg,
    "parallel": parallel,
    "seeds": seeds,
    "n_epochs": n_epochs,
    "batch_size": args.batch_size,
    "epoch_len": args.epoch_len,
}

# Display.
print("-----------------------------------------------------------------")
for key in config:
    print(f"{key}:" + " " * (16 - len(key)), config[key])
print(f"Start:" + " " * 11, {str(datetime.datetime.now())})
print("-----------------------------------------------------------------")

def create_output_path(epsilon, batch_size, base_dir="experiments/results"):
    """
    Creates a safe output path string with proper error handling.
    
    Args:
        epsilon (float): Epsilon value
        batch_size (int): Batch size value
        base_dir (str): Base directory name, defaults to "results"
    
    Returns:
        str: Sanitized output path
    """
    try:
        # Ensure numeric values are valid
        if not isinstance(epsilon, (int, float)) or not isinstance(batch_size, int):
            raise ValueError("Invalid types: epsilon must be numeric and batch_size must be integer")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        # Format the path string, ensuring numbers are formatted cleanly
        # path = os.path.join(base_dir + f"_{float(epsilon):.6f}_{batch_size}")
        path = os.path.join(base_dir + f"_{float(epsilon):.6f}")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        return path
        
    except (TypeError, ValueError) as e:
        print(f"Error creating output path: {str(e)}")
        # Fallback to a safe default with timestamp
        fallback_path = os.path.join(base_dir, f"backup_{int(time.time())}")
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path
    except OSError as e:
        print(f"Error creating directory: {str(e)}")
        # Fallback to current directory if all else fails
        return "."


# Usage in your code:
try:
    output_path = create_output_path(args.epsilon, args.batch_size)
except Exception as e:
    print(f"Unexpected error creating output path: {str(e)}")
    output_path = "results_default"
# Run optimization.
def worker(optim):
    name, lr = optim["optimizer"], optim["lr"]
    diverged = False
    for seed in seeds:
        code = compute_training_curve(
            dataset,
            model_cfg,
            optim,
            seed,
            n_epochs,
            out_path=output_path
        )
        if code == FAIL_CODE:
            diverged = True
    if diverged:
        print(f"Optimizer '{name}' diverged at learning rate {lr}!")


tic = time.time()
if parallel:
    Parallel(n_jobs=args.n_jobs)(delayed(worker)(optim) for optim in optim_cfgs)
else:
    for optim in optim_cfgs:
        worker(optim)
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")

# Save best configuration.
find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds,out_path=output_path)

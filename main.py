import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from argparser import parser

from functools import partial
from bayes_flow import BayesFlow
from training import train_online
from summary_network import SummaryNet

args = parser.parse_args()

if args.data_gen == "old":
    from original_epidemics_model import data_generator
elif args.data_gen == "reimplement":
    from epidemics_model import data_generator
elif args.data_gen == "exact":
    from exact_epidemic_model import data_generator
elif args.data_gen == "vaccine":
    from exact_vaccination_epidemic_model import data_generator

torch.manual_seed(args.seed)
np.random.seed(args.seed)


print("Cuda available: {}".format(torch.cuda.is_available()))
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed(args.seed)

Ns = {"Germany": 83e6,
      "India": 1300e6,
      "Brazil": 211e6,
      "Israel": 9e6}

def load_data(begin_date="2020-01-22", end_date="2021-07-31", country="Germany"):
    # Read data from csv, first columns is index
    # (indexed by dates)
    data = pd.read_csv(f"{args.data}/{country.capitalize()}.csv", index_col=0)
    data = data[begin_date:end_date]
    
    # Don't do the reshape now, but later using `normalise`
    # So we treat the real data just like synthetic data from simulator
    new_I = data["Confirmed"].to_numpy(dtype=np.int64)
    new_R = data["Recovered"].to_numpy(dtype=np.int64)
    new_D = data["Deaths"].to_numpy(dtype=np.int64)
    new_V = data["People_fully_vaccinated"].to_numpy(dtype=np.int64)
    
    if args.data_gen == "vaccine":
        data = np.stack([new_I, new_R, new_D, new_V]).T
    else:
        data = np.stack([new_I, new_R, new_D]).T
    
    if args.max_days is not None:
        data = data[:args.max_days, :]
    
    # If necessary, truncate s.t. T is multiple of 7
    if args.weekly:
        leftover_days = data.shape[0] % 7
        data = data[:-leftover_days, :]
    
        assert data.shape[0] % 7 == 0

    return data


print("Loading Data")
data = load_data(country=args.country.capitalize())
print("Data shape:", data.shape)

mean = np.mean(data, axis=0)[:3]
std = np.std(data, axis=0)[:3]

N = Ns[args.country.capitalize()]
T = data.shape[0] // 7 if args.weekly else data.shape[0]

print("T: {}".format(T))

if args.data_gen == "old":
    data_gen = partial(data_generator, T=T, N=N, mean_g=mean, std_g=std, sim_diff=16)
elif args.data_gen == "reimplement":
    data_gen = partial(data_generator, T=T, N=N, weekly=args.weekly, accept_only_reasonable=True, num_interventions= T // 20)
elif args.data_gen == "exact":
    data_gen = partial(data_generator, T=T, data=data, N=N, mean_g=mean, std_g=std, weekly=args.weekly)
elif args.data_gen == "vaccine":
    data_gen = partial(data_generator, T=T, data=data, N=N, mean_g=mean, std_g=std, weekly=args.weekly)


# Network hyperparameters
hyperparams = {
    'n_blocks': 8,
    'n_layers': 3,
    'n_units': 196,
    'activation': nn.ELU,
    'weight_decay': 0.0
}

N_addin = 1 if args.data_gen == "reimplement" else 0
num_compartments = 3 if args.data_gen == "vaccine" else 3 
in_channels = (7+N_addin) * num_compartments if args.weekly else (1 + N_addin) * num_compartments

channels_per_compartment = 64
x_dim = channels_per_compartment * num_compartments
theta_dim = data_gen(1)['theta'].shape[1]
print(x_dim, theta_dim)

print("Build BaysFlow model")
summary_net = SummaryNet(in_channels, x_dim, num_compartments)
model = BayesFlow(hyperparams, theta_dim, x_dim, summary_net=summary_net, device=device)

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,  weight_decay=hyperparams['weight_decay'])
elif args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=hyperparams['weight_decay'])
elif args.optimizer == "adamw":  # I like AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=hyperparams['weight_decay'])
else:
    print("The optimizer option is not available")

if args.resume:
    print(f"Resuming model from {args.resume}")
    with open(args.resume, "rb") as f:
        model_st, optimizer_st = torch.load(f)#TODO: add lr scheduler?
        model.load_state_dict(model_st)
        optimizer.load_state_dict(optimizer_st)


decay_steps = 1000
decay_rate = .99
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_steps, gamma=decay_rate)

print("\nStarting Training\n")
for ep in range(1, args.epochs+1):
    np.random.seed(args.seed + ep)  # Needed for data multiprocessing
    
    losses = train_online(
        model=model,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        data_gen=data_gen,
        iterations=args.iterations,
        batch_size=args.batch_size,
        device = device,
        num_workers = args.workers
    )
    with open(args.save, "wb") as f:
        torch.save([model.state_dict(), optimizer.state_dict()], f)


#TODO: Visualizations, Evaluations


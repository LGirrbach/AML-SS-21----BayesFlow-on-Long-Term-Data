import time
import argparse

parser = argparse.ArgumentParser(description='BayesFlow')

parser.add_argument(
    '--data', type=str,
    default='data',
    help='Location of data'
    )

parser.add_argument(
    '--country',
    action = 'store',
    type=str,
    choices=['Germany', 'Israel', 'Brazil', 'India'],
    default='Germany',
    help='Country data to use'
    )

parser.add_argument(
    '--weekly',
    action = 'store_true',
    help='Stack weekly data'
    )

parser.add_argument(
    '--max_days',
    action = 'store',
    type=int,
    default = None,
    help = 'Maximum number of days'
    )

parser.add_argument(
    '--lr', 
    action = 'store',
    type=float,
    default=0.0005,
    help='Initial learning rate'
    )

parser.add_argument(
    '--epochs',
    action = 'store',
    type=int,
    default=4,
    help='Upper epoch limit'
    )

parser.add_argument(
    '--iterations',
    action = 'store',
    type=int,
    default=10,
    help='Iterations per epoch'
    )

parser.add_argument(
    '--batch_size',
    action = 'store',
    type=int,
    default=64,
    help='batch size'
    )

parser.add_argument(
    '--seed',
    action = 'store',
    type=int,
    default=118899,
    help='Random seed'
    )

parser.add_argument(
    '--workers',
    action = 'store',
    type=int,
    default=6,
    help='Number of parallel dataset workers'
    )

parser.add_argument(
    '--cuda',
    action = 'store_true',
    help='Whether to use CUDA'
    )

time_info = ''.join(str(time.time()).split('.'))

parser.add_argument(
    '--save',
    action = 'store',
    type=str,
    default=time_info+'_checkpoint.pt',
    help='Path to save the model to'
    )

parser.add_argument(
    '--resume',
    action = 'store',
    type=str, 
    default='',
    help='If given: Path to checkpoint to resume from'
    )

parser.add_argument(
    '--optimizer',
    action = 'store',
    type=str,
    choices=['sgd', 'adam', 'adamw'],
    default='adam',
    help='Optimizer options: sgd, adam'
    )

parser.add_argument(
    '--lr-scheduler',
    action = 'store',
    type=str,
    choices=['steplr'],
    default='steplr',
    help='LR scheduler options: steplr'
    ) #TODO:

parser.add_argument(
    '--data_gen',
    action = 'store',
    type=str,
    choices=['old', 'exact', 'vaccine', 'reimplement'],
    default='exact',
    help = "Data Generation: old, exact, vaccine, reimplement"
    )

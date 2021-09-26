import torch
import numpy as np

from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


class SimulationDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_gen, batch_size, iterations):
        self.batch_size = batch_size
        self.data_gen = data_gen
        self.iterations = iterations
    
    def __iter__(self):
        for _ in range(self.iterations):
            yield self.data_gen(self.batch_size)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train_online(model, optimizer, lr_scheduler, data_gen, iterations,
                 batch_size, clip_value=5., device="cpu", seed=None,
                 num_workers=6):
    """
    Trains given model for 1 epoch = `iterations` steps.
    
    :param model:     BayesFlow model
    :param optimizer: Model optimizer
    :lr_scheduler:    Learning rate scheduler
    :data_gen:        Callable that generates batches of simulated data
    :iterations:      Number of iterations to train
    :batch_size:      Batch size
    :clip_value:      Max. norm of gradient
    """
    # Prepare a dict for storing losses
    losses = []
    running_loss = 0.0

    model = model.to(device)
    
    num_workers = num_workers
    dataset = SimulationDataset(data_gen, batch_size, iterations // num_workers)
    dataloader = DataLoader(dataset, batch_size=None,
                            num_workers=num_workers, worker_init_fn=worker_init_fn)

    # Run training loop
    #train_iterator = trange(iterations)
    train_iterator = tqdm(enumerate(dataloader), total=(iterations // num_workers)*num_workers)
    for it, batch in train_iterator:
        # Generate inputs for the network
        #batch = data_gen(batch_size)
        z, log_jac_det = model(
            batch['theta'].to(device),
            batch['x'].to(device)
            )
        
        optimizer.zero_grad()
        
        # Calculate loss 
        loss = 0.5*torch.sum(z**2, 1) - log_jac_det
        loss = loss.mean() / z.shape[0]
        loss.backward()
        
        # Perform gradient descent
        clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        lr_scheduler.step()

        # Store losses
        detached_loss = loss.detach().cpu().item()
        losses.append(detached_loss)
        running_loss = 0.95 * running_loss + 0.05 * detached_loss
        
        # Display running loss
        train_iterator.set_postfix_str(
            "Iteration: {}, Loss: {:.2f}, Running Loss: {:.2f}" \
                .format(it+1, 100*detached_loss, 100*running_loss)
        )

    return losses

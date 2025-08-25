"""Functions for seeding."""

import torch
import numpy as np
import random
import os

def seed_worker(worker_id):
    """Seed for DataLoaders."""
    worker_seed = torch.initial_seed() % 2**32    
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def seed_everything(SEED):
    """Seed in general."""
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    torch.manual_seed(SEED)
   # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
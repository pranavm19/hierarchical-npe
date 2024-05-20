import torch
import numpy as np

import BioMime.utils.basics as bm_basics
import BioMime.models.generator as bm_gen

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = bm_basics.update_config('../BioMime/BioMime/configs/config.yaml')

"""
Define the simulators for the local (6-parameter) and global+local (7-parameter)
BioMime models. 

"""

def initialize_generator(model_path, num_conds=None):
    if num_conds is not None:
        config['Model']['Generator']['num_conds'] = num_conds
    generator = bm_gen.Generator(config.Model.Generator)
    generator = bm_basics.load_generator(model_path, generator, 'cuda')
    generator = generator.to(device)
    return generator

def sample_biomime(generator, pars):
    if pars.ndim == 1:
        pars = pars[None, :]

    n_MU = pars.shape[0]
    sim_muaps = []

    for _ in range(10):
        cond = pars.to(device)
        sim = generator.sample(n_MU, cond.float(), cond.device)

        sim = sim.to("cpu")
        if n_MU == 1:
            sim = sim.permute(1, 2, 0).detach()
        else:
            sim = sim.permute(0, 2, 3, 1).detach()
        sim_muaps.append(sim)

    muap = np.array(sim_muaps).mean(0)
    
    return torch.from_numpy(muap.flatten())

# Initialize the generators
BIOMIME6 = initialize_generator('biomime_weights/model_linear.pth')
BIOMIME7 = initialize_generator('biomime_weights/biomime7_weights.pth', num_conds=7)

# Define simulators
def simulator_biomime6(pars):
    return sample_biomime(BIOMIME6, pars)

def simulator_biomime7(pars):
    return sample_biomime(BIOMIME7, pars)

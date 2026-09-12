import torch

# The reference path runs many very small CPU tensor ops. On a Della node torch
# defaults to one intra-op thread per core (80 here), and the OpenMP fork/join
# overhead dominates by ~400x on these shapes. Pin to a single thread so the
# reference suite runs in seconds instead of minutes.
torch.set_num_threads(1)

# %%
import torch
curr = torch.tensor([[0], [0], [0]], dtype = torch.float32)
curr[0,0] += 1
states, action, rewards = curr
curr

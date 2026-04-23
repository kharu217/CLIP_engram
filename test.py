import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange
import torch.nn as nn

A = torch.tensor([1, 3, 5, 2, 100, 2, 3], dtype=torch.float16)
B = F.gumbel_softmax(A, hard=True)
print(B)

to_dis_token = nn.Sequential(
    # using a conv layer instead of a linear one -> performance gains
    nn.Conv2d(3, 128, kernel_size=32, stride=32),
    Rearrange('b e (h) (w) -> b (h w) e'),
)
temp = torch.randn((10, 3, 224, 224))
tokens = torch.argmax(nn.functional.gumbel_softmax(to_dis_token(temp), hard=True), dim=2)
print(tokens[0, :])
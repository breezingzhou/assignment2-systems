# %%
import torch
from torch import nn
# %%


class ToyModel(nn.Module):
  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    print(f"after fc1 {x.dtype}")
    x = self.relu(x)
    x = self.ln(x)
    print(f"after ln {x.dtype}")
    x = self.fc2(x)
    print(f"after fc2 {x.dtype}")
    return x


# %%
model = ToyModel(5, 2).cuda()
x = torch.randn(3, 5, dtype=torch.float32).cuda()

for dtype in [torch.float16, torch.bfloat16]:
  print(f"\n=== Testing dtype: {dtype} ===")
  with torch.autocast(device_type="cuda", dtype=dtype):
    print(f"model.params: {model.fc1.weight.dtype}")
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"loss dtype: {loss.dtype}")
    print(f"Grad dtype: {model.fc1.weight.grad.dtype}")  # type: ignore

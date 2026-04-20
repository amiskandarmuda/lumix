# lumix

`lumix` is a small JAX/Flax library for optical neural networks.

It is organized around concrete Linen layers:

- `ClementsLinear`
- `WilliamsonNonlinearity`
- `PowerReadout`

The pure mathematical kernels live under `lumix.functional`.

## Example

```python
from flax import linen as nn
from lumix.linen.clements import ClementsLinear
from lumix.linen.readout import PowerReadout
from lumix.linen.williamson import WilliamsonNonlinearity


class ElectroOpticNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = ClementsLinear(width=16)(x)
        x = WilliamsonNonlinearity(train_gain=True)(x)
        x = ClementsLinear(width=16)(x)
        x = WilliamsonNonlinearity(train_gain=True)(x)
        x = PowerReadout(classes=self.classes)(x)
        return x
```

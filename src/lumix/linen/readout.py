from flax import linen as nn

from lumix.functional.readout import channel_power, class_probs


class PowerReadout(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        return class_probs(channel_power(values), self.classes)

from flax import linen as nn

from lumix.functional.readout import class_logits, class_probs, intensity


class ProbabilityReadout(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        return class_probs(intensity(values), self.classes)


class LogitReadout(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, values):
        return class_logits(intensity(values), self.classes)

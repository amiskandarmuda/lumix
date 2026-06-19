"""Reference-exact training algorithms for Lumix."""

from lumix.training.directional import DirectionalDerivativeResult, bp_dd_step, directional_derivative_gradient
from lumix.training.forward_forward import (
    ff_ad_step,
    ff_dd_step,
    ffzero_margin_loss,
    ffzero_onn_simplex_loss,
)
from lumix.training.insitu import (
    InSituStepResult,
    clements_square_law_logits,
    insitu_classification_step,
    insitu_mse_gradients,
)
from lumix.training.physical import PhysicalMappingResult, unitary_linear_to_clements_params

__all__ = [
    "DirectionalDerivativeResult",
    "InSituStepResult",
    "PhysicalMappingResult",
    "bp_dd_step",
    "clements_square_law_logits",
    "directional_derivative_gradient",
    "ff_ad_step",
    "ff_dd_step",
    "ffzero_margin_loss",
    "ffzero_onn_simplex_loss",
    "insitu_classification_step",
    "insitu_mse_gradients",
    "unitary_linear_to_clements_params",
]

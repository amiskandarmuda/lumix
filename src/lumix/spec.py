from dataclasses import dataclass

from flax import struct


@struct.dataclass
class ClementsSpec:
    width: int = struct.field(pytree_node=False)
    depth: int = struct.field(pytree_node=False)
    perm: object
    mask: object


@dataclass(frozen=True)
class MeshSpec:
    width: int
    depth: int = 1
    mode: str = "pair"
    hadamard: bool = False


@dataclass(frozen=True)
class NonlinearitySpec:
    tap: float = 0.1
    gain: float = 0.05 * 3.141592653589793
    bias: float = 1.0 * 3.141592653589793
    train_gain: bool = False
    train_bias: bool = False


@dataclass(frozen=True)
class TrainSpec:
    epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 5e-3

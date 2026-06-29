"""Convert Lumix optical matrices into Tidy3D inverse-design templates."""

from lumix.inverse_design.layers import LayerSelector
from lumix.inverse_design.matrix import from_matrix_array, from_matrix_file
from lumix.inverse_design.specs import (
    CurvedContainerSpec,
    DesignParameterSet,
    DeviceDesignSpec,
    MatrixObjectiveSpec,
    PassivityPolicy,
    PortTaperSpec,
    PortCounts,
    TopologyRegionSpec,
)
from lumix.inverse_design.template import InverseDesignTemplate, from_lumix_checkpoint

__all__ = [
    "CurvedContainerSpec",
    "DesignParameterSet",
    "DeviceDesignSpec",
    "InverseDesignTemplate",
    "LayerSelector",
    "MatrixObjectiveSpec",
    "PassivityPolicy",
    "PortTaperSpec",
    "PortCounts",
    "TopologyRegionSpec",
    "from_lumix_checkpoint",
    "from_matrix_array",
    "from_matrix_file",
]

from typing import TypeAlias, Sequence, Annotated
from numpy.typing import NDArray
import numpy as np

# Basic numerical types
Real: TypeAlias = float | np.float64
Complex: TypeAlias = complex | np.complex128

# General vector and matrix types
FloatVector: TypeAlias = NDArray[np.float64]  # Shape: (n,)
ComplexVector: TypeAlias = NDArray[np.complex128]  # Shape: (n,)
Float2D: TypeAlias = NDArray[np.float64]  # Shape: (n, m)
Complex2D: TypeAlias = NDArray[np.complex128]  # Shape: (n, m)

# Molecular structure types (keeping these as they're fundamental)
AtomicCoordinates: TypeAlias = NDArray[np.float64]  # Shape: (n_atoms, 3)
AtomicNumbers: TypeAlias = NDArray[np.int32]  # Shape: (n_atoms,)

# Type for atomic symbols with validation
AtomicSymbol: TypeAlias = Annotated[str, "Valid chemical element symbol"]

# Energy types with units (keeping these as they're fundamental)
Energy: TypeAlias = Annotated[float, "Energy in Hartree"]
EnergyEV: TypeAlias = Annotated[float, "Energy in electron volts"]

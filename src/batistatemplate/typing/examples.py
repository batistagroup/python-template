from typing import Annotated

import numpy as np
from numpy.typing import NDArray

# Basic numerical types
type Real = float | np.float64
type Complex = complex | np.complex128

# General vector and matrix types
type FloatVector = NDArray[np.float64]  # Shape: (n,)
type ComplexVector = NDArray[np.complex128]  # Shape: (n,)
type Float2D = NDArray[np.float64]  # Shape: (n, m)
type Complex2D = NDArray[np.complex128]  # Shape: (n, m)

type AtomicCoordinates = NDArray[np.float64]  # Shape: (n_atoms, 3)
type AtomicNumbers = NDArray[np.int32]  # Shape: (n_atoms,)

# Type for atomic symbols with validation
type AtomicSymbol = Annotated[str, "Valid chemical element symbol"]

# Energy types with units (keeping these as they're fundamental)
type Energy = Annotated[float, "Energy in Hartree"]
type EnergyEV = Annotated[float, "Energy in electron volts"]

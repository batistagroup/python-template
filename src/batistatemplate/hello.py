from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence, TypeVar, Generic, Callable, cast
import numpy as np
from numpy.typing import NDArray

from batistatemplate.utils.logging_config import logger
from batistatemplate.typing.examples import (
    AtomicCoordinates,
    AtomicNumbers,
    Energy,
    EnergyEV,
    AtomicSymbol,
    Real,
    FloatVector,
    Float2D
)

class UnitSystem(Enum):
    """Enumeration of supported unit systems."""
    ATOMIC = auto()      # Atomic units (Bohr, Hartree)
    STANDARD = auto()    # Standard units (Angstrom, eV)
    SI = auto()         # SI units (meter, Joule)

@dataclass(frozen=True)
class Atom:
    """Immutable dataclass representing an atom."""
    symbol: AtomicSymbol
    position: FloatVector
    atomic_number: int
    mass: Real

    @classmethod
    def from_symbol(cls, symbol: AtomicSymbol, position: FloatVector, /) -> "Atom":
        """Create an Atom instance from a symbol and position."""
        atomic_data = {
            'H': (1, 1.008),
            'He': (2, 4.003),
            'Li': (3, 6.941),
            'Be': (4, 9.012),
            'B': (5, 10.811),
            'C': (6, 12.011),
            'N': (7, 14.007),
            'O': (8, 15.999),
        }
        
        if symbol not in atomic_data:
            logger.error(f"Attempted to create atom with unknown chemical element: {symbol}")
            raise ValueError(f"Unknown chemical element: {symbol}")
            
        atomic_number, mass = atomic_data[symbol]
        logger.debug(f"Created atom {symbol} at position {position}")
        return cls(symbol, position, atomic_number, mass)

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Generic result container with optional error handling."""
    value: T
    success: bool = True
    error_message: str | None = None

class Molecule:
    """Class representing a molecular system."""
    
    def __init__(self, atoms: Sequence[Atom], unit_system: UnitSystem = UnitSystem.ATOMIC) -> None:
        self.atoms = tuple(atoms)  # Make immutable
        self.unit_system = unit_system
        logger.info(f"Created molecule with {len(atoms)} atoms using {unit_system.name} unit system")
        
    @property
    def coordinates(self) -> AtomicCoordinates:
        """Get atomic coordinates as numpy array."""
        return np.array([atom.position for atom in self.atoms])
    
    @property
    def atomic_numbers(self) -> AtomicNumbers:
        """Get atomic numbers as numpy array."""
        return np.array([atom.atomic_number for atom in self.atoms], dtype=np.int32)
    
    @property
    def center_of_mass(self) -> FloatVector:
        """Calculate center of mass of the molecule."""
        masses = np.array([atom.mass for atom in self.atoms])
        weighted_coords = self.coordinates * masses[:, np.newaxis]
        return cast(FloatVector, np.sum(weighted_coords, axis=0) / np.sum(masses))
    
    def calculate_distance_matrix(self) -> Float2D:
        """Calculate matrix of interatomic distances."""
        coords = self.coordinates
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return cast(Float2D, np.sqrt(np.sum(diff * diff, axis=-1)))
    
    def nuclear_repulsion(self) -> Result[Energy]:
        """Calculate nuclear repulsion energy with error handling."""
        try:
            distance_matrix = self.calculate_distance_matrix()
            atomic_numbers = self.atomic_numbers
            
            # Create charge matrix
            charges = atomic_numbers[:, np.newaxis] * atomic_numbers[np.newaxis, :]
            
            # Zero the diagonal to avoid self-interaction
            np.fill_diagonal(distance_matrix, np.inf)
            
            # Calculate energy
            energy = 0.5 * np.sum(charges / distance_matrix)
            logger.debug(f"Calculated nuclear repulsion energy: {energy:.6f} atomic units")
            return Result(energy)
            
        except Exception as e:
            logger.error(f"Failed to calculate nuclear repulsion energy: {str(e)}")
            return Result(0.0, success=False, error_message=str(e))

def apply_to_coordinates(
    func: Callable[[AtomicCoordinates], AtomicCoordinates],
    molecule: Molecule,
    /
) -> Molecule:
    """Higher-order function to transform molecular coordinates."""
    new_coords = func(molecule.coordinates)
    new_atoms = [
        Atom(a.symbol, np.array(pos, dtype=np.float64), a.atomic_number, a.mass)
        for a, pos in zip(molecule.atoms, new_coords)
    ]
    return Molecule(new_atoms, molecule.unit_system)

def translate_molecule(
    molecule: Molecule,
    /,
    *,
    vector: FloatVector,
) -> Molecule:
    """Translate molecule by a vector.
    
    Args:
        molecule: The molecule to translate
        vector: Translation vector (x, y, z)
    
    Returns:
        New molecule with translated coordinates
    """
    logger.debug(f"Translating molecule by vector {vector}")
    return apply_to_coordinates(
        lambda coords: coords + vector,
        molecule
    )

def rotate_molecule(
    molecule: Molecule,
    /,
    *,
    angle: float,
    axis: FloatVector = np.array([0, 0, 1]),
) -> Molecule:
    """Rotate molecule around an axis by given angle.
    
    Args:
        molecule: The molecule to rotate
        angle: Rotation angle in radians (positive = counterclockwise)
        axis: Rotation axis vector (default: z-axis)
    
    Returns:
        New molecule with rotated coordinates
    """
    logger.debug(f"Rotating molecule by {angle:.2f} radians around axis {axis}")
    def rotation_matrix(theta: float, axis_vec: FloatVector) -> Float2D:
        """Generate 3D rotation matrix using Rodrigues' rotation formula."""
        # Normalize the axis vector
        axis_vec = axis_vec / np.linalg.norm(axis_vec)
        
        # Build the cross product matrix
        cross_matrix = np.array([
            [0, -axis_vec[2], axis_vec[1]],
            [axis_vec[2], 0, -axis_vec[0]],
            [-axis_vec[1], axis_vec[0], 0]
        ], dtype=np.float64)
        
        # Rodrigues' rotation formula with corrected sign
        result = (np.eye(3) + np.sin(-theta) * cross_matrix +
                (1 - np.cos(-theta)) * (cross_matrix @ cross_matrix))
        return cast(Float2D, result)
    
    rot_mat = rotation_matrix(angle, axis)
    return apply_to_coordinates(
        lambda coords: coords @ rot_mat.T,
        molecule
    )

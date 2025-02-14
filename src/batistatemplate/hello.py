from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar, cast

import numpy as np

from batistatemplate.typing.examples import (
    AtomicCoordinates,
    AtomicNumbers,
    AtomicSymbol,
    Energy,
    Float2D,
    FloatVector,
    Real,
)
from batistatemplate.utils.logging_config import logger


class UnitSystem(Enum):
    """Supported unit systems."""

    ATOMIC = auto()  # Atomic units (Bohr, Hartree)
    STANDARD = auto()  # Standard units (Angstrom, eV)
    SI = auto()  # SI units (meter, Joule)


@dataclass(frozen=True)
class Atom:
    """Represents an atom."""

    symbol: AtomicSymbol
    position: FloatVector
    atomic_number: int
    mass: Real

    @classmethod
    def from_symbol(cls, symbol: AtomicSymbol, position: FloatVector, /) -> "Atom":
        """Creates an Atom instance from a chemical symbol and position.

        Args:
            symbol: The chemical symbol of the atom (e.g., "H", "He").
            position: The 3D coordinates of the atom as a FloatVector.

        Returns:
            An Atom instance.

        Raises:
            ValueError: If the chemical symbol is unknown.
        """
        atomic_data = {
            "H": (1, 1.008),
            "He": (2, 4.003),
            "Li": (3, 6.941),
            "Be": (4, 9.012),
            "B": (5, 10.811),
            "C": (6, 12.011),
            "N": (7, 14.007),
            "O": (8, 15.999),
        }

        if symbol not in atomic_data:
            logger.error(f"Attempted to create atom with unknown chemical element: {symbol}")
            raise ValueError(f"Unknown chemical element: {symbol}")

        atomic_number, mass = atomic_data[symbol]
        logger.debug(f"Created atom {symbol} at position {position}")
        return cls(symbol, position, atomic_number, mass)


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """A generic container for function results, including error handling.

    Type parameter ``T`` represents the type of the ``value`` attribute,
    i.e., the type of the successful result.
    """

    value: T
    success: bool = True
    error_message: str | None = None


class Molecule:
    """Represents a molecular system."""

    def __init__(self, atoms: Sequence[Atom], unit_system: UnitSystem = UnitSystem.ATOMIC) -> None:
        """Initializes a Molecule instance.

        Args:
            atoms: A sequence of Atom objects representing the atoms in the molecule.
            unit_system: The unit system to use for the molecule (default: UnitSystem.ATOMIC).
        """
        self.atoms = tuple(atoms)  # Make immutable
        self.unit_system = unit_system
        logger.info(f"Created molecule with {len(atoms)} atoms using {unit_system.name} unit system")

    @property
    def coordinates(self) -> AtomicCoordinates:
        """Atomic coordinates as a numpy array.

        Returns:
            A numpy array of shape (n_atoms, 3) containing the atomic coordinates.
        """
        return np.array([atom.position for atom in self.atoms])

    @property
    def atomic_numbers(self) -> AtomicNumbers:
        """Atomic numbers as a numpy array.

        Returns:
            A numpy array of shape (n_atoms,) containing the atomic numbers.
        """
        return np.array([atom.atomic_number for atom in self.atoms], dtype=np.int32)

    @property
    def center_of_mass(self) -> FloatVector:
        """Calculates the center of mass of the molecule.

        Returns:
            A FloatVector representing the center of mass coordinates.
        """
        masses = np.array([atom.mass for atom in self.atoms])
        weighted_coords = self.coordinates * masses[:, np.newaxis]
        return cast(FloatVector, np.sum(weighted_coords, axis=0) / np.sum(masses))

    def calculate_distance_matrix(self) -> Float2D:
        """Calculates the matrix of interatomic distances.

        Returns:
            A Float2D numpy array of shape (n_atoms, n_atoms) where each element (i, j)
            is the distance between atom i and atom j.
        """
        coords = self.coordinates
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return cast(Float2D, np.sqrt(np.sum(diff * diff, axis=-1)))

    def nuclear_repulsion(self) -> Result[Energy]:
        """Calculates the nuclear repulsion energy.

        This method calculates the Coulomb repulsion energy between all pairs of atomic nuclei in the molecule.
        It includes error handling and returns a Result object.

        Returns:
            A Result object containing the nuclear repulsion energy (in atomic units)
            if successful, or an error message if the calculation fails.
        """
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


def apply_to_coordinates(func: Callable[[AtomicCoordinates], AtomicCoordinates], molecule: Molecule, /) -> Molecule:
    """Applies a coordinate transformation function to a molecule.

    This is a higher-order function that takes a function which transforms atomic coordinates
    and applies it to the coordinates of the given molecule.

    Args:
        func: A callable that takes AtomicCoordinates and returns transformed AtomicCoordinates.
        molecule: The Molecule object to be transformed.

    Returns:
        A new Molecule object with transformed coordinates.
    """
    new_coords = func(molecule.coordinates)
    new_atoms = [
        Atom(a.symbol, np.array(pos, dtype=np.float64), a.atomic_number, a.mass)
        for a, pos in zip(molecule.atoms, new_coords, strict=False)
    ]
    return Molecule(new_atoms, molecule.unit_system)


def translate_molecule(
    molecule: Molecule,
    /,
    *,
    vector: FloatVector,
) -> Molecule:
    """Translates a molecule by a given vector.

    Args:
        molecule: The molecule to translate.
        vector: The translation vector (x, y, z).

    Returns:
        A new Molecule object that is translated by the given vector.
    """
    logger.debug(f"Translating molecule by vector {vector}")
    return apply_to_coordinates(lambda coords: coords + vector, molecule)


def rotate_molecule(
    molecule: Molecule,
    /,
    *,
    angle: float,
    axis: FloatVector | None = None,
) -> Molecule:
    """Rotates a molecule around an axis by a given angle.

    Args:
        molecule: The molecule to rotate.
        angle: The rotation angle in radians (positive for counterclockwise rotation).
        axis: The rotation axis vector. Defaults to the z-axis ([0, 0, 1]).

    Returns:
        A new Molecule object that is rotated by the given angle around the given axis.
    """
    if axis is None:
        axis = np.array([0, 0, 1])

    logger.debug(f"Rotating molecule by {angle:.2f} radians around axis {axis}")

    def rotation_matrix(theta: float, axis_vec: FloatVector) -> Float2D:
        """Generates a 3D rotation matrix using Rodrigues' rotation formula.

        Args:
            theta: The rotation angle in radians (positive for counterclockwise).
            axis_vec: The rotation axis vector.

        Returns:
            A 3x3 Float2D numpy array representing the rotation matrix.
        """
        # Normalize the axis vector
        axis_vec = axis_vec / np.linalg.norm(axis_vec)

        # Build the cross product matrix
        cross_matrix = np.array(
            [[0, -axis_vec[2], axis_vec[1]], [axis_vec[2], 0, -axis_vec[0]], [-axis_vec[1], axis_vec[0], 0]],
            dtype=np.float64,
        )

        # Rodrigues' rotation formula with corrected sign
        result = np.eye(3) + np.sin(-theta) * cross_matrix + (1 - np.cos(-theta)) * (cross_matrix @ cross_matrix)
        return cast(Float2D, result)

    rot_mat = rotation_matrix(angle, axis)
    return apply_to_coordinates(lambda coords: coords @ rot_mat.T, molecule)

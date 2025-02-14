import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from batistatemplate.hello import (
    Atom,
    Molecule,
    UnitSystem,
    rotate_molecule,
    translate_molecule,
)


@pytest.fixture  # type: ignore
def water_molecule() -> Molecule:
    """Create a water molecule in a typical geometry."""
    atoms = [
        Atom.from_symbol("O", np.array([0.0, 0.0, 0.0])),
        Atom.from_symbol("H", np.array([0.0, 0.757, 0.587])),
        Atom.from_symbol("H", np.array([0.0, -0.757, 0.587])),
    ]
    return Molecule(atoms)


@pytest.fixture  # type: ignore
def methane_molecule() -> Molecule:
    """Create a methane molecule in a tetrahedral geometry."""
    # Tetrahedral angles
    a = 1.0 / np.sqrt(3)  # cos(109.47°/2)
    atoms = [
        Atom.from_symbol("C", np.array([0.0, 0.0, 0.0])),
        Atom.from_symbol("H", np.array([a, a, a])),
        Atom.from_symbol("H", np.array([-a, -a, a])),
        Atom.from_symbol("H", np.array([-a, a, -a])),
        Atom.from_symbol("H", np.array([a, -a, -a])),
    ]
    return Molecule(atoms)


class TestAtom:
    """Test suite for the Atom class."""

    def test_atom_creation(self) -> None:
        """Test direct atom creation and factory method."""
        # Test direct creation
        atom = Atom("H", np.array([0.0, 0.0, 0.0]), 1, 1.008)
        assert atom.symbol == "H"
        assert atom.atomic_number == 1
        assert_almost_equal(atom.mass, 1.008)

        # Test factory method
        atom2 = Atom.from_symbol("O", np.array([1.0, 1.0, 1.0]))
        assert atom2.symbol == "O"
        assert atom2.atomic_number == 8
        assert_almost_equal(atom2.mass, 15.999)

    def test_atom_immutability(self) -> None:
        """Test that Atom instances are immutable."""
        atom = Atom.from_symbol("H", np.array([0.0, 0.0, 0.0]))
        with pytest.raises(dataclasses.FrozenInstanceError):
            atom.symbol = "He"

    def test_invalid_symbol(self) -> None:
        """Test error handling for invalid atomic symbols."""
        with pytest.raises(ValueError):
            Atom.from_symbol("Xx", np.array([0.0, 0.0, 0.0]))


class TestMolecule:
    """Test suite for the Molecule class."""

    def test_molecule_creation(self, water_molecule: Molecule) -> None:
        """Test molecule creation and basic properties."""
        assert len(water_molecule.atoms) == 3
        assert water_molecule.unit_system == UnitSystem.ATOMIC

        # Test coordinates property
        coords = water_molecule.coordinates
        assert coords.shape == (3, 3)
        assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])

    def test_center_of_mass(self, water_molecule: Molecule) -> None:
        """Test center of mass calculation."""
        com = water_molecule.center_of_mass
        # COM should be closer to O than H due to mass difference
        assert_array_almost_equal(com[2], 0.0652, decimal=3)

    def test_distance_matrix(self, water_molecule: Molecule) -> None:
        """Test interatomic distance calculation."""
        dist_mat = water_molecule.calculate_distance_matrix()
        assert dist_mat.shape == (3, 3)
        # Check symmetry
        assert_array_almost_equal(dist_mat, dist_mat.T)
        # Check diagonal
        assert_array_almost_equal(np.diagonal(dist_mat), 0.0)

    def test_nuclear_repulsion(self, water_molecule: Molecule) -> None:
        """Test nuclear repulsion energy calculation."""
        result = water_molecule.nuclear_repulsion()
        assert result.success
        assert result.value > 0
        assert result.error_message is None


class TestMolecularOperations:
    """Test suite for molecular operations."""

    def test_translation(self, methane_molecule: Molecule) -> None:
        """Test molecular translation."""
        vector = np.array([1.0, 2.0, 3.0])
        translated = translate_molecule(methane_molecule, vector=vector)

        # Check that original molecule is unchanged
        assert_array_almost_equal(methane_molecule.coordinates[0], np.array([0.0, 0.0, 0.0]))

        # Check that new molecule is translated
        assert_array_almost_equal(translated.coordinates[0], vector)

    def test_rotation(self, water_molecule: Molecule) -> None:
        """Test molecular rotation."""
        # Rotate 180° around z-axis
        rotated = rotate_molecule(water_molecule, angle=np.pi)

        # Check that H atoms are inverted in xy plane
        original_h1 = water_molecule.coordinates[1][:2]  # xy coordinates of first H
        rotated_h1 = rotated.coordinates[1][:2]
        assert_array_almost_equal(rotated_h1, -original_h1)

        # z-coordinates should remain unchanged
        assert_array_almost_equal(water_molecule.coordinates[:, 2], rotated.coordinates[:, 2])

    def test_rotation_custom_axis(self, water_molecule: Molecule) -> None:
        """Test rotation around a custom axis."""
        # Rotate 90° around y-axis
        rotated = rotate_molecule(water_molecule, angle=np.pi / 2, axis=np.array([0.0, 1.0, 0.0]))

        # Original x coordinate should become z coordinate
        assert_array_almost_equal(water_molecule.coordinates[0][0], -rotated.coordinates[0][2])

    def test_successive_operations(self, methane_molecule: Molecule) -> None:
        """Test that operations can be chained."""
        # Translate then rotate
        translated = translate_molecule(methane_molecule, vector=np.array([1.0, 0.0, 0.0]))
        final = rotate_molecule(translated, angle=np.pi / 2, axis=np.array([0.0, 1.0, 0.0]))

        # Check final position of central carbon
        assert_array_almost_equal(final.coordinates[0], np.array([0.0, 0.0, 1.0]))

    def test_invalid_operations(self, water_molecule: Molecule) -> None:
        """Test that operations fail gracefully with invalid arguments."""
        # Test with wrong argument style
        with pytest.raises(TypeError):
            translate_molecule(water_molecule, np.array([1.0, 0.0, 0.0]))

        with pytest.raises(TypeError):
            rotate_molecule(water_molecule, np.pi)  # missing keyword

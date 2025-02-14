import numpy as np

from batistatemplate.hello import (
    Atom,
    Molecule,
    rotate_molecule,
    translate_molecule,
)

if __name__ == "__main__":
    water = Molecule(
        [
            Atom.from_symbol("O", np.array([0.0, 0.0, 0.0])),
            Atom.from_symbol("H", np.array([0.0, 0.757, 0.587])),
            Atom.from_symbol("H", np.array([0.0, -0.757, 0.587])),
        ]
    )

    print("Initial water molecule coordinates:")
    print(water.coordinates)
    print("\nCenter of mass:")
    print(water.center_of_mass)

    # Calculate and print the distance matrix
    print("\nInteratomic distances (in atomic units):")
    print(water.calculate_distance_matrix())

    # Calculate nuclear repulsion energy
    nuc_rep = water.nuclear_repulsion()
    if nuc_rep.success:
        print(f"\nNuclear repulsion energy: {nuc_rep.value:.6f} atomic units")
    else:
        print(f"Error calculating nuclear repulsion: {nuc_rep.error_message}")

    # Demonstrate molecular transformations
    # 1. Translate the molecule
    translated = translate_molecule(water, vector=np.array([1.0, 1.0, 1.0]))
    print("\nTranslated coordinates:")
    print(translated.coordinates)

    # 2. Rotate the molecule 90 degrees around y-axis
    rotated = rotate_molecule(water, angle=np.pi / 2, axis=np.array([0.0, 1.0, 0.0]))
    print("\nRotated coordinates (90° around y-axis):")
    print(rotated.coordinates)

    # Create a methane molecule with tetrahedral geometry
    a = 1.0 / np.sqrt(3)  # cos(109.47°/2)
    methane = Molecule(
        [
            Atom.from_symbol("C", np.array([0.0, 0.0, 0.0])),
            Atom.from_symbol("H", np.array([a, a, a])),
            Atom.from_symbol("H", np.array([-a, -a, a])),
            Atom.from_symbol("H", np.array([-a, a, -a])),
            Atom.from_symbol("H", np.array([a, -a, -a])),
        ]
    )

    print("\nMethane molecule interatomic distances:")
    print(methane.calculate_distance_matrix())

# Hello module

This module provides a modern, type-safe implementation of molecular structure manipulation tools. It serves as an example of Python 3.12+ features and best practices in scientific computing.

## Key Features

- Strong type hints with custom types
- Immutable data structures
- Vectorized operations using NumPy
- Error handling with generic result types
- Modern API design patterns

### Core Classes

#### `UnitSystem` (Enum)

Think of enums as a way to create a fixed set of allowed options in your code. Let's look at different approaches:

```python
# Approach 1: String validation
def calculate_energy(value, unit_system: str):
    unit_system = unit_system.lower()
    valid_units = {"atomic", "standard", "si"}
    if unit_system not in valid_units:
        raise ValueError(f"Invalid unit system. Must be one of: {valid_units}")

    if unit_system == "atomic":
        return value * 27.211396
    elif unit_system == "standard":
        return value
    else:  # si
        return value * 1.602176634e-19

# Problems with this approach:
# 1. No autocomplete - you have to remember valid options
# 2. No type checking until runtime
# 3. String validation logic repeated in every function
# 4. If you rename "atomic" to "au", you need to find and update every string
```

Using enums solves these problems:

```python
class UnitSystem(Enum):
    ATOMIC = auto()
    STANDARD = auto()
    SI = auto()

def calculate_energy(value, unit_system: UnitSystem):
    if unit_system == UnitSystem.ATOMIC:
        return value * 27.211396
    elif unit_system == UnitSystem.STANDARD:
        return value
    else:  # UnitSystem.SI
        return value * 1.602176634e-19

# Benefits:
# 1. IDE autocomplete shows all options
# 2. Type checking catches errors before running the code
# 3. No need for validation - invalid options won't compile
# 4. Rename UnitSystem.ATOMIC to UnitSystem.AU and IDE can update all uses
```

The `auto()` function just handles the behind-the-scenes numbering so you don't have to. It's like having Python automatically number your enum options (1, 2, 3...) instead of you doing it manually.

#### `Atom` (Immutable Dataclass)

Dataclasses are perfect for "data containers" - objects that mainly hold data with little or no behavior. They automatically generate common methods like `__init__`, `__repr__`, and `__eq__`. Here's why we use a dataclass for `Atom`:

```python
# Without dataclass - lots of boilerplate
class AtomOldStyle:
    def __init__(self, symbol, position, atomic_number, mass):
        self.symbol = symbol
        self.position = position
        self.atomic_number = atomic_number
        self.mass = mass

    def __eq__(self, other):
        if not isinstance(other, AtomOldStyle):
            return False
        return (self.symbol == other.symbol and
                self.position == other.position and
                self.atomic_number == other.atomic_number and
                self.mass == other.mass)

    def __repr__(self):
        return f"Atom(symbol={self.symbol}, position={self.position}, ...)"

# With dataclass - same functionality, much cleaner
@dataclass(frozen=True)
class Atom:
    symbol: AtomicSymbol
    position: FloatVector
    atomic_number: int
    mass: Real
```

We use `frozen=True` to make the dataclass immutable - once an atom is created, its properties can't be changed. This prevents bugs from accidental modifications and makes the code easier to reason about.

#### `Molecule` (Class)

Unlike `Atom`, `Molecule` needs to be a regular class because it has complex behavior beyond just storing data. Here's why:

```python
# Molecule needs to be a regular class because it:
class Molecule:
    def __init__(self, atoms: Sequence[Atom], unit_system: UnitSystem = UnitSystem.ATOMIC):
        self._atoms = tuple(atoms)  # Make immutable
        self._unit_system = unit_system

    @property
    def coordinates(self) -> np.ndarray:
        """Complex calculation of coordinates."""
        return np.array([atom.position for atom in self._atoms])

    def calculate_distance_matrix(self) -> np.ndarray:
        """Complex geometric calculations."""
        coords = self.coordinates
        # ... complex matrix calculations ...

    def nuclear_repulsion(self) -> float:
        """Physics calculations involving multiple atoms."""
        # ... complex energy calculations ...

# Use a regular class when you need:
# 1. Complex initialization logic
# 2. Methods that perform calculations
# 3. Properties that compute values on demand
# 4. Control over data access and modification
```

**Rule of thumb:**

- Use dataclasses when your class is primarily about storing data (like `Atom`)
- Use regular classes when you need complex behavior or calculations (like `Molecule`)
- Use frozen dataclasses when you want immutability

### Parameter Passing Conventions

The module uses modern Python parameter passing patterns to create a clear and safe API:

1. **Positional-Only Parameters (`/`)**
    - Used for the primary object being operated on (molecule)
    - Makes code more concise when the parameter name is obvious
    - Example: `translate_molecule(molecule, /, *, vector)`

2. **Keyword-Only Parameters (`*`)**
    - Used for operation parameters with meaningful names
    - Makes function calls self-documenting
    - Prevents parameter order confusion
    - Example: `rotate_molecule(mol, *, angle=np.pi, axis=[0,0,1])`

### Available Operations

#### Translation

Usage:

```python
# Good - clear what the vector represents
translated = translate_molecule(mol, vector=[1.0, 0.0, 0.0])

# Error - must use keyword argument
translated = translate_molecule(mol, [1.0, 0.0, 0.0])  # TypeError
```

#### Rotation

Usage:

```python
# Good - clear what each parameter means
rotated = rotate_molecule(mol, angle=np.pi/2, axis=[0, 1, 0])

# Also good - using default z-axis
rotated = rotate_molecule(mol, angle=np.pi)

# Error - must use keyword arguments
rotated = rotate_molecule(mol, np.pi/2, [0, 1, 0])  # TypeError
```

### Design Philosophy

The module follows five core principles that guide its implementation:

1. **Type Safety**: Leverages Python's type system to catch errors at compile time

    ```python
    def rotate_molecule(molecule: Molecule, /, *, angle: float) -> Molecule:
        """Type hints ensure correct usage at development time."""
    ```

2. **Immutability**: Prevents accidental state modifications

    ```python
    @dataclass(frozen=True)  # Makes instances immutable
    class Atom:
        symbol: AtomicSymbol
        position: FloatVector

    # Operations return new instances instead of modifying
    new_mol = translate_molecule(mol, vector=[1.0, 0.0, 0.0])
    ```

3. **Error Handling**: Makes failures explicit and informative

    ```python
    def parse_xyz_file(path: str) -> Result[Molecule]:
        """Returns Result type to handle both success and failure cases."""
        try:
            return Result(value=read_molecule(path))
        except FileNotFoundError:
            return Result(value=None, success=False,
                        error_message="File not found")
    ```

4. **Clean API**: Prioritizes clarity and prevents usage errors

    ```python
    # Function definition enforces calling convention
    def rotate_molecule(molecule: Molecule, /, *, angle: float, axis: FloatVector):
        """/ makes 'molecule' positional-only
            * makes remaining args keyword-only"""

    # In actual use, just call with keywords - clean and clear
    rotated = rotate_molecule(
        water_molecule,    # positional argument
        angle=np.pi/2,     # keyword arguments
        axis=[0, 1, 0]
    )
    ```

### Examples

#### Creating and Manipulating Molecules

```python
# Create water molecule
atoms = [
    Atom.from_symbol('O', [0.0, 0.0, 0.0]),
    Atom.from_symbol('H', [0.0, 0.757, 0.587]),
    Atom.from_symbol('H', [0.0, -0.757, 0.587])
]
water = Molecule(atoms)

# Translate and rotate
translated = translate_molecule(water, vector=[1.0, 0.0, 0.0])
rotated = rotate_molecule(
    translated,
    angle=np.pi/2,
    axis=[0.0, 1.0, 0.0]
)

# Calculate properties
com = rotated.center_of_mass
distances = rotated.calculate_distance_matrix()
energy = rotated.nuclear_repulsion()
```

### Casting Numpy Results

When working with numpy operations in a statically typed Python codebase, we often need to explicitly cast the results using `typing.cast()`. This is because numpy's type system and Python's static type hints don't always align perfectly. Numpy operations typically return `numpy.ndarray` with dynamic shape and dtype information that mypy cannot infer at compile time. For example, when we perform operations like `np.sum()` or matrix multiplication, the static type checker cannot automatically determine that the result matches our custom type aliases like `FloatVector` or `Float2D`. By using `cast()`, we provide an explicit guarantee to the type checker that the result conforms to our expected type, while maintaining runtime type safety through numpy's own type system.

#### How `cast()` Works

It's important to understand that `typing.cast()` is purely a type checker instruction - it performs no runtime checking or conversion. Here's what that means in practice:

```python
from typing import cast

# This works - we're telling the type checker "trust me, this is a FloatVector"
result = np.array([1.0, 2.0, 3.0])  # Type is ndarray[Any, dtype[float64]]
vector: FloatVector = cast(FloatVector, result)  # Type checker is happy

# This also "works" - cast doesn't check at runtime!
bad_cast = cast(FloatVector, "not an array")  # Type checker is happy but this is wrong
# The error will only show up when you try to use bad_cast as an array

# The right way is to ensure your code actually produces the correct type:
def calculate_vector() -> FloatVector:
    result = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    # We know this will be a 1D float64 array, but the type checker doesn't
    return cast(FloatVector, result)
```

Think of `cast()` as a promise to the type checker: "I know this value will have the right type at runtime." It's the developer's responsibility to ensure this promise is kept. This is why we use it carefully and only in situations where we're certain about the types, like numpy operations where we know the shape and dtype of the result.

## Source Code

::: batistatemplate.hello
handler: python
options:
show_root_heading: true
show_source: true
members: - Atom - Molecule - UnitSystem - apply_to_coordinates - translate_molecule - rotate_molecule

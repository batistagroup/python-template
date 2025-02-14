# Testing Scientific Code

This guide covers principles and practices for testing scientific code, with a focus on computational chemistry, machine learning, and numerical computing.

## Why Test Scientific Code?

1. **Correctness Verification**
    - Scientific code often implements complex mathematical formulas
    - Small errors can propagate and lead to incorrect results
    - Tests help verify mathematical and physical principles

2. **Reproducibility**
    - Tests document expected behavior
    - Help ensure consistent results across different environments
    - Critical for scientific reproducibility

3. **Code Evolution**
    - Safe refactoring of performance-critical code
    - Confidence when updating dependencies
    - Protection against regression

## What to Test

### 1. Physical Invariants

```python
def test_rotation_preserves_distances():
    """Test that rotation preserves interatomic distances."""
    dist_before = molecule.calculate_distance_matrix()
    rotated = rotate_molecule(molecule, angle=np.pi/3)
    dist_after = rotated.calculate_distance_matrix()
    assert_array_almost_equal(dist_before, dist_after)
```

### 2. Mathematical Properties

- Symmetry relations
- Conservation laws
- Matrix properties (hermiticity, unitarity)
- Boundary conditions

### 3. Edge Cases

- Zero values
- Extreme values
- Singular matrices
- Empty/single-element inputs

### 4. Error Handling

- Invalid inputs
- Numerical instabilities
- Resource limitations
- Type mismatches

## Testing ML Code

### 1. Data Pipeline

- Data loading and preprocessing
- Augmentation correctness
- Batch generation
- Edge cases in data

```python
def test_data_normalization():
    """Test that normalization preserves relative ordering."""
    x_raw = np.random.randn(100, 10)
    x_norm = normalize_features(x_raw)

    # Check bounds
    assert np.all((-1 <= x_norm) & (x_norm <= 1))

    # Check relative ordering is preserved
    for i in range(10):
        correlation = np.corrcoef(x_raw[:, i], x_norm[:, i])[0, 1]
        assert abs(correlation) > 0.99
```

### 2. Model Components

- Layer operations
- Activation functions
- Loss calculations
- Gradient computations

```python
def test_softmax_properties():
    """Test mathematical properties of softmax."""
    logits = np.random.randn(10, 5)
    probs = softmax(logits)

    # Sum to 1
    assert_array_almost_equal(np.sum(probs, axis=1), 1.0)

    # Range [0, 1]
    assert np.all((0 <= probs) & (probs <= 1))
```

### 3. Training Process

- Learning rate scheduling
- Weight updates
- Convergence on simple cases
- Reproducibility with fixed seeds

### 4. Model Behavior

- Invariance properties
- Expected behavior on synthetic data
- Performance bounds
- Resource usage

## Testing Numerical Code

### 1. Numerical Stability

```python
def test_numerical_stability():
    """Test stability of operations on poorly conditioned matrices."""
    # Create ill-conditioned matrix
    A = create_ill_conditioned_matrix(condition_number=1e10)
    x = np.random.randn(A.shape[1])

    # Test different solution methods
    x1 = solve_direct(A, b)
    x2 = solve_iterative(A, b)

    # Check both methods give similar results
    assert_array_almost_equal(x1, x2, decimal=5)
```

### 2. Precision and Tolerance

- Use appropriate tolerances for floating-point comparisons
- Consider platform differences
- Test with different precisions
- Handle underflow/overflow

## Testing Strategies

### 1. Property-Based Testing

```python
@given(st.lists(st.floats(min_value=-1e3, max_value=1e3), min_size=1))
def test_normalization_properties(x):
    """Test normalization for arbitrary inputs."""
    normalized = normalize_vector(x)
    assert abs(np.linalg.norm(normalized) - 1.0) < 1e-10
```

### 2. Parameterized Tests

```python
@pytest.mark.parametrize("angle", [0, np.pi/2, np.pi, 3*np.pi/2])
def test_rotation_special_angles(angle):
    """Test rotation for special angles."""
```

### 3. Reference Solutions

- Compare against known analytical solutions
- Test against established software
- Verify with different algorithms

## Best Practices

1. **Test Organization**
    - Group related tests in classes
    - Use fixtures for common setups
    - Separate unit and integration tests

2. **Performance Considerations**
    - Keep unit tests fast
    - Use smaller datasets for tests
    - Mark slow tests appropriately

3. **Documentation**
    - Document test assumptions
    - Explain physical/mathematical meaning
    - Reference equations or papers

4. **Continuous Integration**
    - Run tests on multiple platforms
    - Test with different dependency versions
    - Include performance benchmarks

## Common Pitfalls

1. **Floating-Point Comparisons**

   ```python
   # Bad
   assert x == 1.0

   # Good
   assert_almost_equal(x, 1.0, decimal=6)
   ```

2. **Random Numbers**

   ```python
   # Bad
   x = np.random.randn(100)

   # Good
   rng = np.random.default_rng(seed=42)
   x = rng.standard_normal(100)
   ```

3. **Resource Management**

   ```python
   # Bad
   def test_large_computation():
       result = compute_with_large_matrix(1000000)

   # Good
   @pytest.mark.large
   def test_large_computation():
       result = compute_with_large_matrix(1000)
   ```

## When to Skip Testing

While testing is crucial, some cases might not require tests:

1. **Visualization Code**
    - Plot formatting
    - Color schemes
    - Interactive features

2. **Configuration**
    - Static configuration files
    - Environment settings
    - Documentation

3. **Prototype Code**
    - Exploratory analysis
    - One-off scripts
    - Temporary debugging code

## Conclusion

Testing scientific code requires a balance between:

- Mathematical correctness
- Numerical stability
- Performance
- Maintainability

Focus on testing the critical aspects that ensure scientific validity and reproducibility of your results.

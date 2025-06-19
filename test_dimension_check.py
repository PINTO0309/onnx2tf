#!/usr/bin/env python3

# Check what permutation would be generated for dimension swap
# For shapes [1,2,1,256,32,1] vs [1,1,1,1,2,1]

shape1 = [1,2,1,256,32,1]
shape2 = [1,1,1,1,2,1]

print("Shape 1:", shape1)
print("Shape 2:", shape2)

# Find non-1 values in shape2
non_one_indices = [i for i, v in enumerate(shape2) if v != 1]
print("\nShape 2 has non-1 values at indices:", non_one_indices)

# The actual mismatch is:
# shape1[1] = 2, shape2[1] = 1 (broadcast ok)
# shape1[4] = 32, shape2[4] = 2 (mismatch!)
# So we need to move shape1[1] to position 4

print("\nAnalysis:")
print("shape1[1] = 2 (matches shape2[4] = 2)")
print("shape1[4] = 32 (doesn't match shape2[4] = 2)")
print("\nRequired permutation: [0, 4, 2, 3, 1, 5]")
print("This moves dimension 1 to position 4 and dimension 4 to position 1")

# Verify the permutation
import numpy as np
arr = np.array(shape1)
perm = [0, 4, 2, 3, 1, 5]
transposed = arr[perm]
print("\nAfter permutation [0,4,2,3,1,5]:")
print("Original shape1:", shape1)
print("Transposed:", transposed.tolist())
print("Target shape2:", shape2)
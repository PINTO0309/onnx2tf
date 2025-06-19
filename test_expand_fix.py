#!/usr/bin/env python3
"""Test the correct permutation for custom_spo2 Expand operation"""

# Error: Dimensions must be equal, but are 32 and 2
# Shape 1: [1,2,1,256,32,1] 
# Shape 2: [1,1,1,1,2,1]

# The error is saying dimension 4 of shape1 (32) doesn't match dimension 4 of shape2 (2)
# But shape1[1] = 2 which DOES match shape2[4] = 2
# So we need to move shape1's dimension 1 to position 4

# The correct permutation is [0, 4, 2, 3, 1, 5]
# This moves:
# - dim 0 stays at 0
# - dim 1 moves to 4 
# - dim 2 stays at 2
# - dim 3 stays at 3
# - dim 4 moves to 1
# - dim 5 stays at 5

shape1 = [1, 2, 1, 256, 32, 1]
perm = [0, 4, 2, 3, 1, 5]

# Apply permutation
transposed_shape = [shape1[perm[i]] for i in range(6)]

print("Original shape1:", shape1)
print("Permutation:", perm) 
print("Transposed shape:", transposed_shape)
print("Target shape2:", [1, 1, 1, 1, 2, 1])
print("\nDoes transposed[4] == 2?", transposed_shape[4] == 2)

# Let's also check what the ExpandFixer would detect
shape2 = [1, 1, 1, 1, 2, 1]
mismatched_dims = []
for i in range(6):
    if shape1[i] != shape2[i] and shape2[i] != 1:
        mismatched_dims.append((i, shape1[i], shape2[i]))

print("\nMismatched dimensions detected by current logic:")
print(mismatched_dims)
print("This is empty because shape2[i] != 1 only at position 4, but shape1[4]=32 != shape2[4]=2")

# The real issue is we need to find where in shape1 we have the value that matches shape2[4]
print("\nFinding where shape1 has value 2:")
for i, val in enumerate(shape1):
    if val == 2:
        print(f"  shape1[{i}] = {val}")

print("\nSo we need to move dimension 1 to position 4!")
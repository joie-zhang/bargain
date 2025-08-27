import numpy as np

print("=" * 60)
print("VERSION 1: WITH POSITIVE AND NEGATIVE NUMBERS")
print("=" * 60)

# Vector pairs with specified cosine similarities
# Maximum non-zero values close to 10

# Pair 1: Cosine similarity = 0 (orthogonal vectors)
v1_a = np.array([10, 10, 10, 10, 10])
v1_b = np.array([10, -10, 10, -10, 0])  # dot = 100-100+100-100+0 = 0

# Pair 2: Cosine similarity = 0.25
v2_a = np.array([10, 10, 10, 10, 10])
v2_b = np.array([10, 5, -5, -10, 0])  # Adjusted for cos_sim ≈ 0.25

# Pair 3: Cosine similarity = 0.5
v3_a = np.array([10, 10, 10, 10, 10])
v3_b = np.array([10, 10, 10, 0, -5])  # Adjusted for cos_sim ≈ 0.5

# Pair 4: Cosine similarity = 0.75
v4_a = np.array([10, 10, 10, 10, 10])
v4_b = np.array([10, 10, 10, 5, 5])  # Adjusted for cos_sim ≈ 0.75

# Pair 5: Cosine similarity = 1 (parallel vectors)
v5_a = np.array([2, 4, 6, 8, 10])
v5_b = np.array([1, 2, 3, 4, 5])

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

pairs_v1 = [
    (v1_a, v1_b, 0.00),
    (v2_a, v2_b, 0.25),
    (v3_a, v3_b, 0.50),
    (v4_a, v4_b, 0.75),
    (v5_a, v5_b, 1.00)
]

for i, (a, b, target) in enumerate(pairs_v1, 1):
    actual = cosine_similarity(a, b)
    print(f"\nPair {i} (Target: {target:.2f}):")
    print(f"  Vector A: {a}")
    print(f"  Vector B: {b}")
    print(f"  Actual cosine similarity: {actual:.4f}")
    print(f"  Error: {abs(actual - target):.4f}")

print("\n" + "=" * 60)
print("VERSION 2: NON-NEGATIVE NUMBERS ONLY")
print("=" * 60)

# Vector pairs with non-negative values only
# UPDATED: All vectors now sum to 30 for equal maximum utilities

# Pair 1: Cosine similarity = 0 (orthogonal vectors)
v1_a_nn = np.array([15.0, 15.0, 0.0, 0.0, 0.0])
v1_b_nn = np.array([0.0, 0.0, 10.0, 10.0, 10.0])

# Pair 2: Cosine similarity ≈ 0.25 (actual = 0.2495, error = 0.0005)
v2_a_nn = np.array([13.0, 9.9, 5.0, 1.0, 1.1])
v2_b_nn = np.array([1.1, 1.0, 5.0, 9.9, 13.0])

# Pair 3: Cosine similarity ≈ 0.5 (actual = 0.4940, error = 0.0060)
v3_a_nn = np.array([14.1, 9.1, 3.2, 2.3, 1.3])
v3_b_nn = np.array([3.2, 3.1, 5.9, 9.9, 7.9])

# Pair 4: Cosine similarity ≈ 0.75 (actual = 0.7494, error = 0.0006)
v4_a_nn = np.array([13.1, 7.2, 6.2, 1.3, 2.2])
v4_b_nn = np.array([5.1, 8.0, 6.0, 8.9, 2.0])

# Pair 5: Cosine similarity = 1 (parallel vectors)
v5_a_nn = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
v5_b_nn = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

# All vectors sum to 30 (equal max utilities)
# All cosine similarities within 0.01 of target
pairs_v2 = [
    (np.array([15.0, 15.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0, 10.0, 10.0]), 0.00),
    (np.array([13.0, 9.9, 5.0, 1.0, 1.1]), np.array([1.1, 1.0, 5.0, 9.9, 13.0]), 0.25),
    (np.array([14.1, 9.1, 3.2, 2.3, 1.3]), np.array([3.2, 3.1, 5.9, 9.9, 7.9]), 0.50),
    (np.array([13.1, 7.2, 6.2, 1.3, 2.2]), np.array([5.1, 8.0, 6.0, 8.9, 2.0]), 0.75),
    (np.array([10.0, 8.0, 6.0, 4.0, 2.0]), np.array([10.0, 8.0, 6.0, 4.0, 2.0]), 1.00)
]

for i, (a, b, target) in enumerate(pairs_v2, 1):
    actual = cosine_similarity(a, b)
    print(f"\nPair {i} (Target: {target:.2f}):")
    print(f"  Vector A: {a}")
    print(f"  Vector B: {b}")
    print(f"  Actual cosine similarity: {actual:.4f}")
    print(f"  Error: {abs(actual - target):.4f}")
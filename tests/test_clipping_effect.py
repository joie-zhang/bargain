#!/usr/bin/env python3
"""Test to demonstrate how clipping affects cosine similarity."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def generate_similar_vector_no_clip(base_vector, target_similarity):
    """Generate vector with target similarity WITHOUT clipping."""
    length = len(base_vector)
    
    # Normalize base vector to unit vector
    v1_norm = base_vector / np.linalg.norm(base_vector)
    
    # Generate a random vector
    random_vec = np.random.uniform(0, 10, length)
    
    # Make it orthogonal to v1_norm using Gram-Schmidt
    v_orthogonal = random_vec - np.dot(random_vec, v1_norm) * v1_norm
    v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal)
    
    # Calculate angle from desired cosine similarity
    angle = np.arccos(np.clip(target_similarity, -1, 1))
    
    # Construct v2 using the angle
    v2_norm = np.cos(angle) * v1_norm + np.sin(angle) * v_orthogonal
    
    # Scale back to original magnitude
    v2 = v2_norm * np.linalg.norm(base_vector)
    
    return v2

def generate_similar_vector_with_clip(base_vector, target_similarity):
    """Generate vector with target similarity WITH clipping."""
    v2 = generate_similar_vector_no_clip(base_vector, target_similarity)
    # Clip to [0, 10]
    v2_clipped = np.clip(v2, 0, 10)
    return v2_clipped

def cosine_similarity(v1, v2):
    """Calculate cosine similarity."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Test with different target similarities
np.random.seed(42)
base_vector = np.array([3.75, 9.51, 7.32, 5.99, 1.56])

print("Effect of Clipping on Cosine Similarity")
print("=" * 60)
print(f"Base vector: {base_vector}")
print()

for target_sim in [0.05, 0.2, 0.5, 0.8, 0.95]:
    print(f"\nTarget Similarity: {target_sim}")
    print("-" * 40)
    
    # Without clipping
    v2_no_clip = generate_similar_vector_no_clip(base_vector, target_sim)
    actual_no_clip = cosine_similarity(base_vector, v2_no_clip)
    
    # With clipping
    v2_clipped = generate_similar_vector_with_clip(base_vector, target_sim)
    actual_clipped = cosine_similarity(base_vector, v2_clipped)
    
    # Check how many values were clipped
    num_clipped = np.sum((v2_no_clip < 0) | (v2_no_clip > 10))
    
    print(f"Without clipping:")
    print(f"  Vector: {v2_no_clip}")
    print(f"  Actual similarity: {actual_no_clip:.4f} (error: {abs(target_sim - actual_no_clip):.4f})")
    
    print(f"With clipping [0, 10]:")
    print(f"  Vector: {v2_clipped}")
    print(f"  Actual similarity: {actual_clipped:.4f} (error: {abs(target_sim - actual_clipped):.4f})")
    print(f"  Values clipped: {num_clipped}/{len(base_vector)}")
import numpy as np

def generate_vectors_with_cosine_similarity(length, target_similarity):
    """
    Generate two vectors of specified length with desired cosine similarity.
    
    Parameters:
    -----------
    length : int
        The length of the vectors to generate
    target_similarity : float
        The desired cosine similarity between the vectors (between -1 and 1)
    
    Returns:
    --------
    tuple : (vector1, vector2)
        Two numpy arrays with the specified cosine similarity
    """
    if not -1 <= target_similarity <= 1:
        raise ValueError("Cosine similarity must be between -1 and 1")
    
    if length < 2:
        raise ValueError("Vector length must be at least 2")
    
    # Generate first vector randomly
    v1 = np.random.randn(length)
    v1 = v1 / np.linalg.norm(v1)  # Normalize to unit vector
    
    # Generate a random orthogonal vector
    random_vec = np.random.randn(length)
    # Make it orthogonal to v1 using Gram-Schmidt
    v_orthogonal = random_vec - np.dot(random_vec, v1) * v1
    v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal)  # Normalize
    
    # Calculate angle from desired cosine similarity
    angle = np.arccos(np.clip(target_similarity, -1, 1))
    
    # Construct v2 using the angle
    v2 = np.cos(angle) * v1 + np.sin(angle) * v_orthogonal
    
    # Scale vectors to have random magnitudes (optional)
    # You can comment these lines out if you want unit vectors
    v1 = v1 * np.random.uniform(1, 10)
    v2 = v2 * np.random.uniform(1, 10)
    
    return v1, v2

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Example usage
if __name__ == "__main__":
    # Define parameters
    vector_length = 10
    desired_similarity = 0.7
    
    # Generate vectors
    vector1, vector2 = generate_vectors_with_cosine_similarity(vector_length, desired_similarity)
    
    # Verify the result
    actual_similarity = cosine_similarity(vector1, vector2)
    
    print(f"Vector 1: {vector1}")
    print(f"Vector 2: {vector2}")
    print(f"Desired cosine similarity: {desired_similarity}")
    print(f"Actual cosine similarity: {actual_similarity:.6f}")
    print(f"Error: {abs(desired_similarity - actual_similarity):.6e}")
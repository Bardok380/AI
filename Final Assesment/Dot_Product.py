def dot_product(vector1, vector2):
    """
    Calculate the dot product of two vectors.
    
    Parameters:
    - vector1: List of numbers
    - vector2: List of numbers

    Returns:
    - Dot product as a number

    Raises:
    - ValueError if vectors are nt the same length
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    
    return sum(x * y for x, y in zip(vector1, vector2))

v1 = [7, 8, 9]
v2 = [4, 5, 6]
result = dot_product(v1, v2)
print("Dot product:", result)
import math

class HypercomplexNumber:
    """
    Represents an N-dimensional hypercomplex number, where the dimension
    is automatically padded to the next power of 2.

    Attributes:
        components (list): A list containing the components of the number.
        dimension (int): The dimension of the number, which is a power of 2.
        p (float): A negative real number parameterizing the algebra.
    """

    def __init__(self, components, p):
        """
        Constructor for the HypercomplexNumber class.

        It takes a list of components, pads it with zeros to ensure
        the length is a power of 2, and stores the parameter p.
        """
        if not isinstance(components, list) or not all(isinstance(c, (int, float)) for c in components):
            raise TypeError("Components must be a list of numeric values.")
        
        if not isinstance(p, (int, float)):
            raise TypeError("Parameter p must be a real number.")
        if p >= 0:
            raise ValueError("Parameter p must be a negative real number.")

        self.p = float(p)

        n = len(components)
        
        if n == 0:
            # If the input list is empty, default to a 1-dimensional number [0]
            target_dim = 1
        else:
            # Calculate the smallest power of 2 that is >= n
            power = math.ceil(math.log2(n)) if n > 0 else 0
            target_dim = 2**power

        # Pad the list with zeros to reach the target dimension
        num_zeros_to_add = target_dim - n
        padded_components = components + [0] * num_zeros_to_add

        self.components = padded_components
        self.dimension = target_dim

    def __repr__(self):
        """
        Returns a human-readable string representation of the object.
        """
        return f"HypercomplexNumber({self.components}, p={self.p})"

    def __eq__(self, other):
        """
        Checks for equality between two HypercomplexNumber objects.
        """
        if not isinstance(other, HypercomplexNumber):
            return False
        # For equality, dimensions, p, and all components must match.
        return self.dimension == other.dimension and self.p == other.p and self.components == other.components

    def __add__(self, other):
        """
        Overloads the '+' operator for addition.
        """
        if not isinstance(other, HypercomplexNumber):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add numbers of different dimensions ({self.dimension} and {other.dimension}).")
        if self.p != other.p:
            raise ValueError(f"Cannot add numbers with different p parameters ({self.p} and {other.p}).")
        
        new_components = [a + b for a, b in zip(self.components, other.components)]
        # The result will have the same dimension and p, so no re-padding is needed.
        return HypercomplexNumber(new_components, self.p)

    def __sub__(self, other):
        """
        Overloads the '-' operator for subtraction.
        """
        if not isinstance(other, HypercomplexNumber):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract numbers of different dimensions ({self.dimension} and {other.dimension}).")
        if self.p != other.p:
            raise ValueError(f"Cannot subtract numbers with different p parameters ({self.p} and {other.p}).")
            
        new_components = [a - b for a, b in zip(self.components, other.components)]
        # The result will have the same dimension and p, so no re-padding is needed.
        return HypercomplexNumber(new_components, self.p)

    def norm(self):
        """
        Calculates the norm of the hypercomplex number.

        The norm is defined as the square root of:
        (sum of squares of even-indexed components) - p * (sum of squares of odd-indexed components).
        """
        sum_even_sq = sum(self.components[i]**2 for i in range(0, self.dimension, 2))
        sum_odd_sq = sum(self.components[i]**2 for i in range(1, self.dimension, 2))
        
        norm_sq = sum_even_sq - self.p * sum_odd_sq
        
        # Norm can be complex if norm_sq is negative. We return the principal root.
        if norm_sq < 0:
            return complex(0, math.sqrt(-norm_sq))
        else:
            return math.sqrt(norm_sq)

# --- Example Usage ---
if __name__ == '__main__':
    # Define a value for p
    p_val = -1.0

    # Example 1: Input with 3 components. Should be padded to dimension 4.
    # Components become [1, 2, 3, 0]
    h1 = HypercomplexNumber([1, 2, 3], p=p_val)
    print(f"Input: [1, 2, 3], p={p_val}")
    print(f"Created Object: {h1}")
    print(f"Dimension: {h1.dimension}")
    # Norm calculation: sqrt((1^2 + 3^2) - (-1)*(2^2 + 0^2)) = sqrt(10 + 4) = sqrt(14)
    print(f"Norm of h1: {h1.norm():.4f}\n") # Expected: sqrt(14) approx 3.7417

    # Example 2: Input with 4 components.
    h2 = HypercomplexNumber([10, 20, 30, 40], p=p_val)
    print(f"Input: [10, 20, 30, 40], p={p_val}")
    print(f"Created Object: {h2}")
    print(f"Dimension: {h2.dimension}")
    # Norm calculation: sqrt((10^2 + 30^2) - (-1)*(20^2 + 40^2)) = sqrt(1000 + 2000) = sqrt(3000)
    print(f"Norm of h2: {h2.norm():.4f}\n") # Expected: sqrt(3000) approx 54.7723
    
    # Example 3: Test addition.
    print("--- Operation Example ---")
    h_sum = h1 + h2
    print(f"h1: {h1}")
    print(f"h2: {h2}")
    print(f"Sum (h1 + h2): {h_sum}\n")
    
    # Example 4: This will raise an error because p values don't match.
    try:
        print("--- Error Handling Example (different p) ---")
        h3 = HypercomplexNumber([5, 5, 5, 5], p=-2.0)
        error_sum = h1 + h3
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")


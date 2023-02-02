import numpy as np

def int_to_bit_strings(int_list, num_bits = -1):

    max_int = max(int_list)

    if num_bits < 0:

        # Get the number of bits needed to represent the largest integer
        num_bits = int(np.ceil(np.log2(max_int + 1)))

    # Initialize an empty matrix with the correct number of rows and columns
    bit_string_matrix = np.empty((len(int_list), num_bits), dtype=int)

    # Fill in the matrix with the bit strings
    for i, integer in enumerate(int_list):
        bit_string = np.binary_repr(integer, width=num_bits)
        bit_string_matrix[i, :] = list(map(int, bit_string))
        
    return bit_string_matrix


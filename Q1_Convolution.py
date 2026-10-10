import numpy as np

# Input matrix
input_matrix = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# 3x3 filter
filter_matrix = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

# Stride and padding
stride = 1
padding = 0

# Calculate output dimensions
input_height, input_width = input_matrix.shape
filter_height, filter_width = filter_matrix.shape

output_height = ((input_height - filter_height + 2 * padding)
                 // stride) + 1

output_width = ((input_width - filter_width + 2 * padding)
                // stride) + 1

# Create output feature map
output = np.zeros((output_height, output_width), dtype=int)

# Perform convolution/filtering
for i in range(0, input_height - filter_height + 1, stride):
    for j in range(0, input_width - filter_width + 1, stride):

        # Select the current region
        region = input_matrix[
            i:i + filter_height,
            j:j + filter_width
        ]

        # Calculate dot product
        output[i // stride, j // stride] = np.sum(
            region * filter_matrix
        )

# Print results
print("Input Matrix:")
print(input_matrix)

print("\nFilter:")
print(filter_matrix)

print("\nOutput Feature Map:")
print(output)

print("\nOutput Shape:")
print(output.shape)

print("\nStride:", stride)
print("Padding:", padding)

print("\nIf stride changes from 1 to 2:")
print("The filter moves two positions at a time.")
print("Therefore, the output feature map becomes smaller.")
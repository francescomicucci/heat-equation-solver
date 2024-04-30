import numpy as np
import matplotlib.pyplot as plt

# Plot the matrix
def plot_matrix(matrix):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='Greys', interpolation='nearest')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def get_index(i, j, k):
    return i + 3*j + 9*k

# Create the matrix
matrix = np.zeros((27, 27))
for k in range(3):
    for j in range(3):
        for i in range(3):
            matrix[get_index(i, j, k), get_index(i, j, k)] = 1
            if i > 0:
                matrix[get_index(i, j, k), get_index(i-1, j, k)]= 1
            if i < 2:
                matrix[get_index(i, j, k), get_index(i+1, j, k)] = 1
            if j > 0:
                matrix[get_index(i, j, k), get_index(i, j-1, k)] = 1
            if j < 2:
                matrix[get_index(i, j, k), get_index(i, j+1, k)] = 1
            if k > 0:
                matrix[get_index(i, j, k), get_index(i, j, k-1)] = 1
            if k < 2:
                matrix[get_index(i, j, k), get_index(i, j, k+1)] = 1

plot_matrix(matrix)

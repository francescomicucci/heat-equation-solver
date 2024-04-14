import matplotlib.pyplot as plt
import pandas as pd
import sys

# Check command line arguments
if len(sys.argv) != 2:
    print("Usage: python3 plotError.py <data.csv>")
    sys.exit(1)

# Get input file name
inputFile = sys.argv[1]

# Load data from csv file
inputData = pd.read_csv(inputFile, sep = ",")

# Plot data
plt.plot(inputData.deltaX, inputData.err, 'b--o', label="error")
plt.plot(inputData.deltaX, inputData.deltaX**1, 'k--', label="deltaX")
plt.plot(inputData.deltaX, inputData.deltaX**2, 'k-', label="deltaX^2")
plt.xlabel('delta X')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
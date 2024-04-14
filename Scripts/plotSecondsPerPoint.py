import matplotlib.pyplot as plt
import pandas as pd
import sys

# Check command line arguments
if len(sys.argv) != 2:
    print("Usage: python3 plotSecondsPerPoint.py <data.csv>")
    sys.exit(1)

# Get input file name
inputFile = sys.argv[1]

# Load data from csv file
inputData = pd.read_csv(inputFile, sep = ",")

# Plot data
plt.plot(inputData.n, inputData.SecondsPerPoint, 'r--o')
plt.xlabel('Number of points')
plt.ylabel('Seconds per point')
plt.xscale('log')
plt.show()
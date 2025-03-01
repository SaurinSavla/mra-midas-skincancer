import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load metadata
file_path = "C:/Data/DJ/azcopydata/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx"
df = pd.read_excel(file_path)

# Plot histograms for numerical columns
numerical_cols = ['midas_age', 'length_(mm)', 'width_(mm)']

df[numerical_cols].hist(figsize=(12, 6), bins=30)
plt.suptitle("Distribution of Numerical Metadata Features", fontsize=14)
plt.show()

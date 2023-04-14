import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a list of years and months
years = list(range(2016, 2023))
months = list(range(1, 13))

# Generate random sales growth values for each year-month combination
data = []
for year in years:
    for month in months:
        if month in [1, 2, 3, 4, 5, 9, 10, 11, 12]:
            # Generate mostly positive growth values
            sales_growth = np.random.uniform(low=1.0, high=5.0)
        else:
            # Generate both positive and negative growth values
            sales_growth = np.random.uniform(low=-0.05, high=1.0)
        data.append([year, month, sales_growth])

# Convert the data to a Pandas DataFrame
columns = ['Year', 'Month', 'Sales Growth']
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('tilak_di_hatti_sales.csv', index=False)

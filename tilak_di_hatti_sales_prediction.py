import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data into a Pandas DataFrame
data = pd.read_csv('tilak_di_hatti_sales.csv')

# print first 10 rows of data 
print('First 10 rows of the dataset:\n', data.head(10))

# Combine the Year and Month columns to create a Date column
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))

# Split the data into training and testing sets
train_data = data[data['Year'] < 2022]
test_data = data[data['Year'] == 2022]

# Prepare the data for training
X_train = train_data[['Year', 'Month']]
y_train = train_data['Sales Growth']

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prepare the data for testing
X_test = test_data[['Year', 'Month']]
y_test = test_data['Sales Growth']

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error: ', rmse)

# Plot the training data and model predictions
plt.figure(figsize=(11, 6))
plt.plot(train_data['Date'], train_data['Sales Growth'], label='Training Data')
plt.plot(test_data['Date'], test_data['Sales Growth'], label='Testing Data')
plt.plot(test_data['Date'], y_pred, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Sales Growth')
plt.title('Sales Growth Prediction')
plt.legend()
plt.show()


# Plot the residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Month'], residuals)
plt.xlabel('Month')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Check the normality of the residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=10)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Normality Check')
plt.show()

# Check the linearity assumption
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Linearity Check')
plt.show()

# Check the homoscedasticity assumption
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Homoscedasticity Check')
plt.show()




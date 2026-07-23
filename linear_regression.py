import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("salary_data.csv")

print(data.head())


# Input and output
X = data[['Experience']]
y = data['Salary']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Predict new salary
years = [[5]]

salary = model.predict(years)

print("Predicted Salary for 5 years experience:", salary[0])


# Graph
plt.scatter(X, y, color="blue")

plt.plot(
    X,
    model.predict(X),
    color="red"
)

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Salary Prediction")

plt.show()
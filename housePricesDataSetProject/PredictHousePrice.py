import pandas as pd



#Loading my selected dataset
dataFrame = pd.read_csv("House Price India.csv")
#See first few rows
print(dataFrame.head())
#Optional: Check data info
print(dataFrame.info())



#Data Cleaning
#Drop rows with missing target or features
dataFrame = dataFrame.dropna(subset=['price'])  # assuming 'price' is your target column

#Fill missing values in numeric columns with median
for col in dataFrame.select_dtypes(include=['float64', 'int64']):
    dataFrame[col].fillna(dataFrame[col].median(), inplace=True)

#Turns text categories into numerical 0/1 columns.
dataFrame = pd.get_dummies(dataFrame, drop_first=True)



#Select Features & Target
# Example: everything except 'price' is a feature
X = dataFrame.drop('price', axis=1)
y = dataFrame['price']



#Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Choose a Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train
model.fit(X_train, y_train)

#Evaluate the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

#Try a Better Model
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("RandomForest R² Score:", r2_score(y_test, y_pred_rf))

#Save the Model
import joblib
joblib.dump(rf_model, "house_price_model.pkl")
#load and reuse model
model = joblib.load("house_price_model.pkl")

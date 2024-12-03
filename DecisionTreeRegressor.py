import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load dataset
data = pd.read_csv('ENB2012_data.csv')

# Define features and targets
X = data.iloc[:, :-2].values  # Assuming all columns except last two are features

y_heating = data.iloc[:, -2].values  # Heating load
y_cooling = data.iloc[:, -1].values  # Cooling load
# Split data
X_train, X_test, y_heating_train, y_heating_test = train_test_split(X, y_heating, test_size=0.2, random_state=42)
_, _, y_cooling_train, y_cooling_test = train_test_split(X, y_cooling, test_size=0.2, random_state=42)

# Train models
heating_model = DecisionTreeRegressor()
heating_model.fit(X_train, y_heating_train)

cooling_model = DecisionTreeRegressor()
cooling_model.fit(X_train, y_cooling_train)



# Serialize models
with open('heating_model.pkl', 'wb') as f:
    pickle.dump(heating_model, f)

with open('cooling_model.pkl', 'wb') as f:
    pickle.dump(cooling_model, f)
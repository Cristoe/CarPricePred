import pandas as pd
df = pd.read_csv('Car_Cleaned_Data.csv')
df.head()
df.columns
#Separate the x and y data
X = df[['Present_Price','Kms_Driven','Car_Age','Fuel_Type_CNG','Fuel_Type_Diesel',
        'Fuel_Type_Petrol','Transmission_Automatic',
        'Transmission_Manual']]
y = df['Selling_Price']
#Fitting the linear regression model to the TRAINING SET
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

import pickle
with open('LR_Model.pkl', 'wb') as Model_file:
    pickle.dump(model, Model_file)

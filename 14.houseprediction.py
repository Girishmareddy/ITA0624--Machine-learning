import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
df = pd.read_csv("C:\\Users\\giris_pu2cvr5\\Downloads\\HousePricePrediction.csv")
df['SalePrice'].fillna(df['SalePrice'].mean(), inplace=True)
X = df.iloc[:, [0, 1, 3, 6, 7, 8,10, 11]]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
new_data = pd.DataFrame({
    'Id': [int(input("Enter Id: "))],
    'MSSubClass': [int(input("Enter MSSubClass: "))],
    'LotArea': [float(input("Enter LotArea: "))],
    'OverallCond': [int(input("Enter OverallCond: "))],
    'YearBuilt': [int(input("Enter YearBuilt: "))],
    'YearRemodAdd': [int(input("Enter YearRemodAdd: "))],
    'BsmtFinSF2': [float(input("Enter BsmtFinSF2: "))],
    'TotalBsmtSF': [float(input("Enter TotalBsmtSF: "))],
})
new_data_for_prediction = new_data[['Id', 'MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF']]
predicted_sale_price = model.predict(new_data_for_prediction)
print("Predicted SalePrice for the New House:", predicted_sale_price[0])


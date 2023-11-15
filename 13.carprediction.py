import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\giris_pu2cvr5\\Downloads\\carprice.csv")
selected_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg', 'price']
df_selected = df[selected_columns]
X = df_selected.drop('price', axis=1)
y = df_selected['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
new_data = pd.DataFrame({
    'wheelbase': [float(input("Enter wheelbase: "))],
    'carlength': [float(input("Enter carlength: "))],
    'carwidth': [float(input("Enter carwidth: "))],
    'carheight': [float(input("Enter carheight: "))],
    'curbweight': [float(input("Enter curbweight: "))],
    'enginesize': [float(input("Enter enginesize: "))],
    'horsepower': [float(input("Enter horsepower: "))],
    'citympg': [float(input("Enter citympg: "))],
    'highwaympg': [float(input("Enter highwaympg: "))],
})

predicted_price = model.predict(new_data)
print(f'Predicted Price: {predicted_price[0]}')







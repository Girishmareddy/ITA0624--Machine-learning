import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("C:\\Users\\giris_pu2cvr5\\Downloads\\position_salaries.csv")
x = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
  
lin.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)

poly.fit(x_poly,y)
lin = LinearRegression()
lin.fit(x_poly,y)

plt.scatter(x,y,color= 'red')

plt.plot(x, lin.predict(poly.fit_transform(x)), color = 'green')
plt.title('Polynomial Regression')
plt.xlabel('Levels')
plt.ylabel('Salary x 10^6')

plt.show()
 

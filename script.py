import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# 1
# print(df.head(5))

# 2
prod_per_year = df.groupby(['year']).totalprod.mean().reset_index()
# print(prod_per_year)

# 3
x = prod_per_year['year'].values.reshape(-1,1)
# print(x)

# 4
y = prod_per_year['totalprod']

# 5
plt.scatter(x, y)

liner = LinearRegression()
liner.fit(x, y)

y_pred = liner.predict(x)

y_pred2 = [x_val*liner.coef_ + liner.intercept_ for x_val in x]

plt.plot(x, y_pred2)

x_future = np.array(range(2013, 2050, 1)).reshape(-1,1)
y_future = liner.predict(x_future)

plt.plot(x_future, y_future)

plt.show()










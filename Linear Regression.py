import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib_inline

path = "C:\\Users\\DELL\\Desktop\\work\\Lib\\homeprices.csv"
df = pd.read_csv(path)
#print(df)

#plt.xlabel('area(sqr feet)')
#plt.ylabel('price(USD $)')
#plt.scatter(df.area, df.price, color = 'red', marker= '+')

##                      CREATING OUR LINEAR REGRESSION MODEL

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

a = reg.intercept_
#print(a)
p = reg.coef_
#print(p)
accuracy= reg.score(X= pd.DataFrame(df["area"]),
                     y= df["price"])
accuracy

train_prediction = reg.predict(X= pd.DataFrame(df["area"]))
train_prediction

from sklearn.metrics import mean_squared_error

RMSE = mean_squared_error(train_prediction, df["price"])**0.5
RMSE
#y = float(input("Enter the area of the house you wish to purchase: "))
#print("Predicted value(price) is =", p * y + a)

df.plot(kind= "scatter",
            x= "area",
            y= "price",
            figsize= (9,9),
            color= "black")

plt.plot(df["area"],
         train_prediction,
         color= "blue")

plt.xlabel('area(sqr feet)', fontsize= 10)
plt.ylabel('price(USD $)', fontsize= 10)
plt.scatter(df.area, df.price, color = 'red', marker= '+')
plt.plot(df.area, reg.predict(df[['area']]), color= 'blue')

##            PREDICTING PRICES INDICATED IN ANOTHER DATASET

path2 ="C:\\Users\\DELL\\Desktop\\work\\Lib\\areas.csv"
d = pd.read_csv(path2)
#print(d.head)

uncover = reg.predict(d)
#print(uncover)
#d['Prices'] = uncover
#print(d.head(5))

#d.to_csv("Price predictions", index= False)
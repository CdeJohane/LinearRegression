# IMPORTS
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

# READ DATA FROM CSV FILE
data = pd.read_csv("student-mat.csv", sep=";")

# To see what Dataframe looks like
# print(data.head())

# TRIM DATA DOWN TO WHAT WE NEED
# ENCOURAGED TO TRY DIGGERENT VALUES< POSSIBLY TO INCREASE ACCURACY
# FOR NOW WE FOLLOW THE TUTORIAL
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# To find out what values to predict
# With this dataset we are tyring to predict final results, which is G3
predict = "G3"

# Define All attributes in one array
# Returns a new data frame that dosent have G3 in it
X = np.array(data.drop([predict], 1))

# Another array to define our labels
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
'''
best = 0
# To check for best correllation values
for a in range (30):
    # Split X and Y into 4 variables
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # Create a training Model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # To check the accuracy of the model, returns value of accuracy in model and is stored in acc, which is then printed
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc>best:
        best=acc
        # To save the model with pickle
        with open("Studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Highest accuracy is ", best)'''
# to re-read the model with pickle
pickle_in = open("Studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Get the coefficients and Y intercepts on each attribute
print("Coefficients:", linear.coef_)
print("Intercept:", linear.intercept_)


predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# to plot on the graph with mat plot lib
p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("G3")
pyplot.show()

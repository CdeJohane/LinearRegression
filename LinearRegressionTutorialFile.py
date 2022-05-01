# IMPORTS
import pandas as pd
import numpy as np
import sklearn
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

# Split X and Y into 4 variables
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
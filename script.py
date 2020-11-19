# set Path variable to identify where the files are
# you need to change it to your working path
path = "C:/Users/user/Documents/dev/python/challenges/DigitRecognizer"

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import itertools
import seaborn as sns

from sklearn.model_selection import train_test_split


# Read Data from CSV File
train = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")

Y_train = train['label']
X_train = train.drop(labels = ["label"], axis = 1)

del train

g = sns.countplot(Y_train)

# Reduce size
# Basically, reducing the numbers to 0-1.
X_train = X_train / 255.0
test = test / 255.0

# Spliting data
train_X, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)

# Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 1000, max_depth = 10, random_state = 2)

model.fit(train_X, Y_train)

test_pred = model.predict(test)

# Making Results and CSV File to eventual submission
results = pd.DataFrame(test_pred)
results.index.name='ImageId'
results.index+=1
results.columns=['Label']
results.to_csv(f'{path}/results.csv', header = True)

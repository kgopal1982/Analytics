#https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
from sklearn import datasets
import xgboost as xgb

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# =============================================================================
# In order for XGBoost to be able to use our data, 
# we’ll need to transform it into a specific format that XGBoost can handle. 
# That format is called DMatrix. It’s a very simple one-linear to 
# transform a numpy array of data to DMatrix format
# =============================================================================
D_train = xgb.DMatrix(X_train, label = y_train)
D_test = xgb.DMatrix(X_test, label = y_test)

# =============================================================================
# max_depth (maximum depth of the decision trees being trained), 
# objective (the loss function being used), and 
# num_class (the number of classes in the dataset)
# 
# The eta can be thought of more intuitively as a learning rate. 
# Rather than simply adding the predictions of new trees to the ensemble with full weight, 
# the eta will be multiplied by the residuals being adding to reduce their weight. 
# This effectively reduces the complexity of the overall model.
# =============================================================================

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
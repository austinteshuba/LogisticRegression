import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import log_loss


# Step 1: Get the datq as a dataframe in Pandas.
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv"

df = pd.read_csv(path)

print(df.head())

# Step 2: Mild Preprocessing
# To use the Logistic Regression package, the target variable has to be an integer
df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
df['churn'] = df['churn'].astype('int') # it was given as float which is annoying
print(df.head())

print(df.count())
print(df.shape) # (# of rows, # of columns)

# Step 3: Get our dataset for the package.
# We need the data as an array
X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(df['churn'])

# Step 4: Normalize the data set
# We have to normalize before doing regression, so some values don't impact the results more than others
X = preprocessing.StandardScaler().fit(X).transform(X)

# Step 5: Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)

# Step 6: Start building the model
# C is the inverse of regularization strength. Lower the value, stronger the regularization

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

# This gets the predictions without probabilities
yhat = LR.predict(X_test)

# This gets the predictions with the probabilities. First col is the probability of class 1 (i.e. yes),
# then second is prob of class 0 (i.e. no, or 1-P(class1))
yhat_prob = LR.predict_proba(X_test)

# Step 7: Evaluation.
# First, we can use the simplest metric: the jaccard similarity index

print(jaccard_similarity_score(y_test, yhat))

# The confusion matrix is also a great display.
# Below, the function will plot the matrix given the classes and the matrix

def plotMatrix(cm, classes, normalize=False, title="Confusion Matrix", colormap = plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=colormap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])

print(cnf_matrix)


# Plot non-normalized confusion matrix
plt.figure()
plotMatrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

# Get the f-scores
np.set_printoptions(precision=2)
print (classification_report(y_test, yhat))

# Lastly, we can check logloss.
# This checks the accuracy using the actual probabilities.
print(log_loss(y_test, yhat_prob))


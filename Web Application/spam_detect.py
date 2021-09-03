import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib as jl

df = pd.read_csv('dataset.csv')

# data preparation
df["label"] = df["Category"].map({'ham': 0, 'spam': 1}) # encoding
df = df.drop(["Category"], axis=1)

corpus = df["Message"]
# using the BoW [Bag of Words] concept
vectorizer = CountVectorizer() # initialize count vectorizer
bow_matrix = vectorizer.fit_transform(corpus) # fit the data to the vectorizer
jl.dump(vectorizer, "vectorizer.pkl") # store the vectorizer as a .pkl file

X = bow_matrix  # Training set
y = df['label'] # Target feature

# Training-testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# SVM
clf = svm.SVC() # Support Vector Classifier
clf.fit(X_train, y_train) # fit the data to the model

# store the model as a .pkl file
jl.dump(clf, "model.pkl")


"""
        Interpret the classifier
        -----

        This function can be called separately to see the misclassified samples

import numpy as np

# make predictions
predicted = clf.predict(X_test)


# find misclassified samples
print("Misclassified Samples")
misses = np.where(y_test != predicted)
misclassified = df.iloc[misses]
print(misclassified)
misclassified[misclassified['label'] == 0]. \
    to_csv(r'misclassified_samples/Misclassified Ham Samples.txt', sep=' ')
misclassified[misclassified['label'] == 1]. \
    to_csv(r'misclassified_Samples/Misclassified Spam Samples.txt', sep=' ')

"""
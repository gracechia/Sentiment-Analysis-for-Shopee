# Importing libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pickle

# Read the clean dataset
reviews = pd.read_csv('clean_train.csv')

X = reviews['content_stem']
y = reviews['target']

# Perform train test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with TF-IDF and Logistic Regression
pipe_tvec_lr = Pipeline([
    ('tvec', TfidfVectorizer(stop_words='english')),
    ('lr', LogisticRegression(random_state=42))
])

# Search over the following values of hyperparameters:
pipe_tvec_lr_params = {
    'tvec__max_features': [300],
    'tvec__min_df': [2,3],
    'tvec__max_df': [.9,.95],
    'lr__penalty': ['l2'],
    'lr__C': [.1, 1]
}

# Instantiate GridSearchCV
gs_tvec_lr = GridSearchCV(pipe_tvec_lr,
                          param_grid = pipe_tvec_lr_params,
                          cv=10)

# Fit model on to training data
gs_tvec_lr.fit(X_train, y_train)

# Create a pipeline with TF-IDF Vectorizer and SVC
pipe_tvec_svc = Pipeline([
    ('tvec', TfidfVectorizer(stop_words='english')),
    ('svc', SVC(probability=True, random_state=42))
])

# Search over the following values of hyperparameters:
pipe_tvec_svc_params = {
    'tvec__max_features': [800],
    'tvec__min_df': [2,3],
    'tvec__max_df': [.9,.95],
    'svc__kernel': ['linear'],
    'svc__C': [.1]
}

# Instantiate GridSearchCV
gs_tvec_svc = GridSearchCV(pipe_tvec_svc,
                          param_grid = pipe_tvec_svc_params,
                          cv=10)

# Fit model on to training data
gs_tvec_svc.fit(X_train, y_train)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('tvec_lr', gs_tvec_lr),
                ('tvec_svc', gs_tvec_svc)],
    voting='soft',
    weights=[1,2]
)

# Fit model on to training data
voting_clf.fit(X_train, y_train)

# Saving model to disk
pickle.dump(voting_clf, open('model.pkl','wb'))

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))

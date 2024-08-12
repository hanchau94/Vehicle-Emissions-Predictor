import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score,confusion_matrix, ConfusionMatrixDisplay, \
classification_report,roc_curve, auc,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')

## train_x ,train_y, test_x test_y take from pre_processing step
# Logistic Regression Model
clf = LogisticRegression()
clf.fit(train_x ,train_y)
y_pred = clf.predict(test_x)
cm = confusion_matrix(test_y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
score = cross_val_score(LogisticRegression(), train_x, train_y, cv=10)

# Optimize the model with GridSearchCV
# Create logistic regression model
lr_model = LogisticRegression()

# Define hyperparameters to search over
hyperparameters = {
    'penalty': ['l1', 'l2',None],
    'C': [0.01, 0.1, 1, 10, 100,1000]
}

# Use GridSearchCV to find best hyperparameters
grid_search = GridSearchCV(lr_model, hyperparameters, cv=5)
grid_search.fit(train_x, train_y)

y_pred = grid_search.predict(test_x)
recal_fore = recall_score(test_y, y_pred, average=None)

# Optimize in the pipeline
# Create a pipeline
pipeline = Pipeline([
    ('select', SelectKBest(score_func=f_classif)), # Select top features using ANOVA F-value
    ('scale', StandardScaler()), # Standardize the data
    ('classify', LogisticRegression()) # Classifier
])
para = {
    'select__k':[5,8,10],
    'classify__penalty': ['l1', 'l2',None],
    'classify__C': [0.01, 0.1, 1, 10, 100,1000]
}

grid_search = GridSearchCV(pipeline, para, cv=5)
grid_search.fit(train_x, train_y)

y_pred = grid_search.predict(test_x)
recal_fore = recall_score(test_y, y_pred, average=None)
predict_p =grid_search.predict_proba(test_x)[:,1]
fpr, tpr, thresholds = roc_curve(test_y, predict_p, \
                                     pos_label=1)
precision, recall, thresholds = precision_recall_curve(test_y, predict_p, pos_label=1)

## Optimizing the SVM model
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge'],
    'max_iter':[100,500,1000],
    'dual': [True,False]
}

# create the grid search object
grid_search = GridSearchCV(svm.LinearSVC(), param_grid, cv=3)

# perform the grid search
grid_search.fit(train_x, train_y)


# predict on the test set using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(test_x)

## Random Forest
# Define the hyperparameters to search over
param_dist = {'n_estimators': randint(50, 200),
              'max_depth': [5, 10, 20, None],
              'max_features': ['sqrt', 'log2']}

# Create a Random Forest object
rfc = RandomForestClassifier(random_state=42)

# Perform randomized search to find the best hyperparameters
rand_search = RandomizedSearchCV(estimator=rfc, 
                                 param_distributions=param_dist, 
                                 n_iter=10, 
                                 cv=5, 
                                 random_state=42)
rand_search.fit(train_x, train_y)

## Neural Network 
param_grid = {
    'hidden_layer_sizes': [(5, 3), (10, 5), (20, 10)],
    'activation': ['logistic', 'relu'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 2000],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

# Create an MLPClassifier object
clf = MLPClassifier(random_state=1)

# Create a GridSearchCV object and fit it to the data
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1)
grid_search.fit(train_x, train_y)

# Get the best model and print the results
best_model = grid_search.best_estimator_
y_hat = best_model.predict(test_x)
recal_clf = recall_score(test_y, y_hat, average=None)

# Create a RandomizedSearchCV object and fit it to the data
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,\
                                   cv=5, verbose=1,random_state=42)
random_search.fit(train_x, train_y)

# Get the best model and print the results
best_model_1 = random_search.best_estimator_
y_hat_1 = best_model_1.predict(test_x)
recall_clf_1 = recall_score(test_y, y_hat_1, average=None)

## Naive Bayes
# Define a pipeline to select the best k features and train the model
clf = GaussianNB()
kbest = SelectKBest(score_func=f_classif)
pipeline = Pipeline(steps=[('kbest', kbest), ('clf', clf)])

# Define the grid search parameters
param_grid = {
    'kbest__k': [2, 4, 6, 8, 10],
}

# Create a GridSearchCV object and fit it to the data
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=1)
grid_search.fit(train_x, train_y)

# Get the best model and print the results
best_model = grid_search.best_estimator_
y_hat = best_model.predict(test_x)


## Compare the time when training and predict result for each model

def model(xtrain,ytrain,xtest,ytest,model):
    if model == "logistic":
        clf = Pipeline([
    ('select', SelectKBest(score_func=f_classif,k=8)), 
    ('scale', StandardScaler()), 
    ('classify', LogisticRegression(C=10))])
        
    elif model == "svm":
        clf = svm.LinearSVC(C= 1, dual= False, loss= 'squared_hinge',\
                            max_iter= 1000, penalty= 'l1')
    elif model == "random forest":
        clf = RandomForestClassifier(max_depth=20, max_features= 'log2',\
                                     n_estimators= 142,random_state=42)
        
    elif model == "neural network":
        clf = MLPClassifier(activation= 'logistic', alpha= 0.001,\
                            hidden_layer_sizes= (20, 10), learning_rate_init= 0.001,\
                            max_iter= 1000, solver= 'adam',random_state=1)
    elif model == "naive bayes":
        clf = Pipeline([
    ('select', SelectKBest(score_func=f_classif,k=4)), 
    ('classify', GaussianNB())])
        
    clf.fit(xtrain,ytrain)
    y_pred = clf.predict(xtest)  
    accuracy = accuracy_score(ytest, y_pred)
    return accuracy     

model_name = ["logistic","svm","random forest","neural network","naive bayes"]
model_time = [0]*len(model_name)
for i in range(len(model_name)):
    start = time.time()
    model_train = model(train_x,train_y,test_x,test_y,model=model_name[i])
    end = time.time()
    time_count = end-start
    model_time[i]=round(time_count,2)
dict_time = dict(zip(model_name,model_time))
print(dict_time)
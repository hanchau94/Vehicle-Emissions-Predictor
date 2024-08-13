# Evaluating the CO2 Emission from Gasoline-Powered Light-Duty Vehicles

Welcome to our project focused on assessing CO2 emissions from gasoline-powered light-duty vehicles. In an era where environmental sustainability is paramount, understanding and regulating vehicle emissions is critical. Our project leverages machine learning techniques to analyze vehicle data, predict CO2 emissions, and compare them against predefined thresholds.

## Table of Contents
- [Objectives](#objectives)
- [Models](#models)
- [Dataset](#dataset)
- [Performance](#performance)
  
## Objectives

- **Assess CO2 Emissions:** Predict CO2 emissions from gasoline-powered light-duty vehicles.
- **Compare with Thresholds:** Determine if the emissions exceed predefined thresholds. We specifically calculate the threshold based on the amount of CO2 emissions from a gallon of gasoline, and the average gasoline vehicle on the road today has a fuel economy of about <a href= "https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=P100U8YT.pdf" > ≈ 35.4 kilometers per gallon </a>.

$$\text{CO2 per km} = \frac{\text{CO2 per gallon}}{MPG}= \frac{8.887}{35.4}=251 \text{ gram}$$

- **Support Regulatory Efforts:** Aid in regulatory efforts and promote cleaner transportation practices.

## Models

- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"> **Logistic regression:**</a> For binary classification issues, logistic regression is a straightforward and understandable approach. It can handle a high number of characteristics and can be applied to both linear and nonlinear decision boundaries. The likelihood of each class is also provided via logistic regression, which is helpful in some applications.
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC"> **LinearSVC:** </a> A linear model called a Linear Support Vector Classifier (SVC) seeks out the ideal hyperplane that divides the two classes. It is effective when the data can be separated linearly and is computationally efficient, especially for large datasets.
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier"> **Random Forest:**</a> To enhance classification performance, Random Forest is an ensemble model that blends various decision trees. In comparison to a single decision tree, it can handle nonlinear decision boundaries and is less prone to overfitting. Additionally, Random Forest can deal with irrelevant characteristics and missing values.
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier"> **Neural network:**</a> A neural network is an effective model that can discover intricate patterns and connections in data. It consists of many layers of nodes that alter the input characteristics nonlinearly. Large datasets may be handled by neural networks, which can also be applied to both linear and nonlinear decision limits.
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB" > **Naive Bayes:**</a> Naive Bayes is a straightforward and quick model that excels in classifying text as well as other high-dimensional datasets. It determines the likelihood of each class based on the occurrence of each feature and makes the assumption that the features are independent. Naive Bayes can be applied to problems involving binary and multiple classes in classification.
- However, we also plan to optimize the models by searching for the best parameters using various techniques, such as SelectKBest(), StandardScaler(), GridSearchCV(), or RandomizedSearchCV() from Scikit-learn to help us fine-tune the models to achieve better accuracy and performance.

## Dataset
Datasets provide model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada <a href="https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64">Resource</a>.

## Performance
The accuracy score and time taken for training and prediction varies for each model, and the results are as follows:
- Logistic regression: 98.65% and 0.05 seconds
- Linear SVC: 97.88% and 0.83 seconds
- Random forest: 99.86% and 0.73 seconds
- Neural network: 97.06% and 1.43 seconds
- Naive Bayes: 96.33% and 0.01 seconds

=> The Random Forest model demonstrated the highest accuracy score and f1-scores for both classes, making it the top performer. However, if interpretability is a priority, Logistic Regression also showed strong performance and might be a better option. Additionally, the logistic regression model was the fastest to train and predict, making it a good choice for time efficiency. Considering the trade-offs between accuracy, interpretability, and time efficiency, the logistic regression model could be a suitable option.


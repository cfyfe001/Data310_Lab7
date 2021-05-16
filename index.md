# Lab 7
## Question 1: If we use RandomForest (random_state=310) max_depth=10 and 1000 trees for ranking the importance of the input features the top three features are (in decreasing order).
```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

columns = 'age gender bmi map tc ldl hdl tch ltg glu'.split()
diabetes = datasets.load_diabetes()  
df = pd.DataFrame(diabetes.data, columns=columns)  
y = diabetes.target  

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=310, max_depth=10,n_estimators=1000)

data = df
data = pd.get_dummies(data)
model.fit(data,y)

features = data.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-3:]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```
Using this code, it is clear that the answer is **ltg, bmi, and map.**

## Question 2: For the diabetes dataset you worked on the previous question, apply stepwise regression with add/drop p-values both set to 0.001. The model selected has the following input variables:
```
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01,
                       threshold_out = 0.05, 
                       verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
    
columns = 'age gender bmi map tc ldl hdl tch ltg glu'.split()
diabetes = datasets.load_diabetes()  
df = pd.DataFrame(diabetes.data, columns=columns)  
y = diabetes.target  

result = stepwise_selection(df,y, [], 0.001, 0.001)
result
```
Using this code allows us to see that the answer is **bmi, ltg, and map.**

## Question 3: For the diabetes dataset scale the input features by z-scores and then apply the ElasticNet model with alpha=0.1 and l1_ratio=0.5. If we rank the variables in the decreasing order of the absolute value of the coefficients the top three variables (in order) are
```
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

columns = 'age gender bmi map tc ldl hdl tch ltg glu'.split()
diabetes = datasets.load_diabetes()  
df = pd.DataFrame(diabetes.data, columns=columns)  
y = diabetes.target  

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

Xs = scale.fit_transform(df)

from sklearn import linear_model as lm
model = lm.ElasticNet(alpha=0.1,l1_ratio = 0.5)
model.fit(Xs,y)

v = -np.sort(-np.abs(model.coef_))
for i in range(df.shape[1]):
  print(df.columns[np.abs(model.coef_)==v[i]])
```
Using this code, the answer is **bmi, ltg, and map.**

## Question 4: A k-fold cross-validation can be used to determine the best choice of hyper-parameters from a finite set of choices.
The answer is **true.** Based on our lecture notes and the active coding in class, it is evident that this statement is true.

## Question 5: In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best minimum samples per leaf (in a range between 1 and 25) the optimal pair of hyper-parameters (such as max depth and min leaf samples) is
```
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

model = DecisionTreeClassifier(random_state=1693)
params = [{'max_depth':np.linspace(1,100,100),'min_samples_leaf': np.linspace(0.1,0.5,20)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(X,y)
print(gs_results.best_params_)
print('The best MSE is achieved when: ', np.abs(gs_results.best_score_))
```
I was not able to correctly find this answer. My error dealt with creating a range for the min_samples_leaf.

## Question 6: In this problem consider 10-fold cross-validations and random_state=12345 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best minimum samples per leaf (in a range between 1 and 25) the number of False Negatives is
```
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix as CM
from sklearn.model_selection import KFold

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

model = DecisionTreeClassifier(random_state=12345)
params = [{'max_depth':np.linspace(1,100,100),'min_samples_leaf': np.linspace(0.1,0.5,20)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(X,y)
print(gs_results.best_params_)
print('The best MSE is achieved when: ', np.abs(gs_results.best_score_))

model = DecisionTreeClassifier(random_state=12345, max_depth = 4, min_samples_leaf = 23)
model.fit(X, y)
predicted_classes = model.predict(X)

spc = ['Malignant','Benign']
cm = CM(y,predicted_classes)
pd.DataFrame(cm, columns=spc, index=spc)
```
Using this code gives us the answer **22.**

## Question 7: In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data set with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best minimum samples per leaf (in a range between 1 and 25) the accuracy is about
``` 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

model = DecisionTreeClassifier(random_state=1693)
params = [{'max_depth':np.linspace(1,100,100),'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}]
gs = GridSearchCV(estimator=model,cv=10,scoring='accuracy',param_grid=params)
gs_results = gs.fit(X,y)
print(gs_results.best_params_)
print('The best accuracy is achieved when: ', np.abs(gs_results.best_score_))
```
As a result of the code, the answer is about **89%.**

## Question 8: Maximum depth, minimum node size, and learning rate are all examples of what type of parameters?:
The answer is **hyperparameters** because all of these are aspects which are included in the classification and regression models we have learned about in lecture.

## Question 9: Selecting an appropriate model requires:
**All of these answers are correct** (both determining if your model is linear or nonlinear and if your model is discrete or continuous). Both of these things are vital to understand to correct select the model which would best fit the data.

## Question 10: A good reason for implementing a feature selection technique is:
**We want to create a parsimonious model.** Having a parsimonious model allows us to predict the output more accurately than using all features within a dataset. This gain in predictive power is essential in data science.

## Question 11: The concept of Principal Component Analysis refers to:
**Determining the directions along which we maximize the variance of the input features.** This answer choice is part of the definition of PCA we learned in class, so this choice is correct.

## Question 12: In this problem the input features will be scaled by the z-scores and consider a use a random_state=1234. If you analyze the data with benign/malign tumors from breast cancer data, consider a decision tree with max_depth=10,min_samples_leaf=20 and fit on 9 principal components the number of true positives is
```
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
X = df.values
y = dat.target

model = DecisionTreeClassifier(max_depth=10,min_samples_leaf=20,random_state=1234)

pca = PCA(n_components=9)
Xpca = pca.fit_transform(X)
pca.fit(Xpca)

model.fit(Xpca, y)
predicted_classes = model.predict(Xpca)

spc = ['Malignant','Benign']
cm = CM(y,predicted_classes)
pd.DataFrame(cm, columns=spc, index=spc)
```
Based on the code, the answer is **196.**

# MachineLearningAssignment-1

|Exercise|Responsible| 
|---|---|
|1|Gundars & Benedikt|
|2|Gundars & Benedikt|
|3|Andrei|
|3|Florian & Gianfranco|
|4|Florian & Gianfranco|

# Student names and numbers:
​
The assignments below should be solved and documented as a mini-project that will form the basis for the examination. When solving the exercises it is important that you

document all relevant results and analyses that you have obtained/performed during the exercises
try to relate your results to the theoretical background of the methods being applied.
Feel free to add cells if you need to. The easiest way to convert to pdf is to save this notebook as .html (File-->Download as-->HTML) and then convert this html file to pdf. You can also export as pdf directly, but here you need to watch your margins as the converter will cut off your code (i.e. make vertical code!).
```python
# Import all necessary libraries here
```
Exercise 1: Decision trees
In this exercise we investigate the Boston Housing dataset, which we treat as a classification problem:

```python
from sklearn.datasets import load_boston
data = load_boston()
​
import numpy as np
from sklearn.model_selection import train_test_split
c = np.array([1 if y > np.median(data['target']) else 0 for y in data['target']])
X_train, X_test, c_train, c_test = train_test_split(data['data'], c, random_state=0)
```

Model learning
a) Learn a decision tree using the training data and evaluate its performance on both the training data and the test data. Generate random training/test partitions or varying sizes and analyze how the accuracy results vary (consult the documentation for train_test_split(.)).

Model analysis
b) Display the decision tree learned using the training data.

c) What are the most important features as determined by the learned tree and does, e.g., the choice of top node seem reasonable to you?

d) How does the features deemed most important by the decision tree learner match the generated tree and your understanding of house prices?

Model complexity
e) Try controlling the complexity of the learned decision tree by adjusting the parameters max_depth, min_samples_split, min_samples_leaf

f) Investigate the effect when changing these parameters:

- Visualize (some of) the trees
- Evaluate the performance of the models on both the training data and the test data
g) Try to find good parameter values using cross-validation. How does the obtained parameters match your manual investigation?

Exercise 2: Regression with random forest
For this exercise we will use the nycflights dataset ("flights.csv").

So far, we have only considered how to use decision trees and random forests for classification. However, both algorithms can also be used for regression tasks, as we will see in the exercises below.

Preprocessing
a) Load the data, and consider how you want to handle missing values and categorical variables (you may choose to remove some features entirely). Carefully consider which variables are categorical. Normalize all relevant variables.

b) In the following, we are going to determine which factors cause departure time delays, and try to predict the length of these delays. However, for several departures, a negative delay have been reported. How do you interpret a negative delay? Consider if you want to modify the negative delays in some way.

Regression analysis: Predicting departure time delays
c) Extract the features and the target variable (in this case the departure time delays) from the dataframe. Split the dataset into test and train sets (technically, we ought to have done this before preprocessing. For the sake of simplicity, we do not conform to this best practice in this exercise).

d) Train a decision tree regressor for predicting departure time delays (you might want to experiment with a few different values of the hyperparameters to avoid too much overfitting). Plot the tree, and explain how decision trees can be used for regression analyses.

e) Do a regression analysis as the one above, but using a random forest instead of a single decision tree. Use a grid-search to determine a good set of hyperparameters. When you have found the best model, score your model on the test set. Comment on the result.

f) Plot the feature importances determined by the tree. Which feature is the most important? Do you have any idea as to why? Remove any features which cannot be used to predict departure time delays in any meaningful way, and redo the analysis. Comment on your results.

Regression analysis: Predicting arrival time delays
In the last part of the exercise, we are going to try to predict arrival time delays as a function of departure time delays - it might be of interest to know how large a delay one should expect after the plane has departed from the airport.

g) Train a decision tree or random forest regressor and an OLS to the dataset, and see how well arrival time delay. can be predicted based on departure time delay.

h) Plot the arrival time delays as a function of the departure time delay, and show the predictions from each of the two regressors.

i) Based on the results obtained above, make a plot that extrapolates a little bit in order to predict delays slightly larger than the largest delay found in the dataset. Which model do you think gives the most trustworthy extrapolation?

Exercise 3: SVM
In this exercise we perform character recognition using SVM classifiers. We use the MNIST dataset, which consists of 70000 handwritten digits 0-9 at a resolution of 28x28 pixels. In the cell below, the dataset is loaded and split into 60000 traning and 10000 testing images, and reshaped into the appropriate shape for an SVM classifier.
```python
from keras.datasets import mnist
​
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)
```
The code-snippet below can be used to see the images corresponding to individual digits:
```python
import matplotlib.pyplot as plt
index = 1
​
plt.imshow(x_train[index].reshape(28,28),cmap=plt.cm.gray_r)
plt.show()
```
To make things a little bit simpler (and faster!), we can extract from the data binary subsets, that only contain the data for two selected digits:
```python
import numpy as np
​
digit0=3
digit1=7
x_bin_train=x_train[np.logical_or(y_train==digit0,y_train==digit1)]
y_bin_train=y_train[np.logical_or(y_train==digit0,y_train==digit1)]
​
x_bin_test=x_test[np.logical_or(y_test==digit0,y_test==digit1)]
y_bin_test=y_test[np.logical_or(y_test==digit0,y_test==digit1)]
​
print("The first training datapoint now is: \n")
plt.imshow(x_bin_train[0].reshape(28,28),cmap=plt.cm.gray_r)
plt.show()
print(y_bin_train[0])
```
Training an SVM model
a) Learn different SVM models by varying e.g. the kernel functions and/or the C- and gamma-parameters. For each configuration, determine the time it takes to learn the model, and the accuracy on the test data. Caution: for some configurations, learning here can take a little while (several minutes).

b) Inspect some misclassified cases. Do they correspond to hard to recognize digits (also for the human reader)? (Hint: you can e.g. use the numpy where() function to extract the indices of the test cases that were misclassified: misclass = np.where(test != predictions) )

c) How do results (time and accuracy) change, depending on whether you consider an 'easy' binary task (e.g., distinguishing '1' and '0'), or a more difficult one (e.g., '4' vs. '5').

d) Explain how a binary classifier, such as an SVM, can be applied to a multiclass classification problem, such as recognizing all 10 digits in the MNIST dataset (no coding required in this exercise!).

e) Identify one or several good configurations that give a reasonable combination of accuracy and runtime. Use these configurations to perform a full classification of the 10 classes in the original dataset (after split into train/test). Using sklearn.metrics.confusion_matrix you can get an overview of all combinations of true and predicted labels (see p. 298-299 in Müller & Guido). What does this tell you about which digits are easy, and which ones are difficult to recognize, and which ones are most easily confused?

Cheating
We next investigate the capability of the different learning approaches to find a good model, knowing that a very accurate model exists. For this, we add a 'cheat column' to our data: we add an additional column to the data matrix that simply contains a 0/1 encoding of the actual class label:
```python
#adding cheating information to the training data:
cheatcol_train=np.array(y_bin_train) #making a copy of the original target array
cheatcol_train[cheatcol_train==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_train[cheatcol_train==digit1]=1
cheatcol_train=cheatcol_train.reshape(-1,1) #getting the dimensions right for the following .hstack operation to work ... 
x_bin_cheat_train = np.hstack((x_bin_train,cheatcol_train))
​
#adding cheating information to the training data:
cheatcol_test=np.array(y_bin_test) #making a copy of the original target array
cheatcol_test[cheatcol_test==digit0]=0  #re-coding the two classes as 0s and 1s
cheatcol_test[cheatcol_test==digit1]=1
cheatcol_test=cheatcol_test.reshape(-1,1) #getting the dimensions right for the following .hstack operation to work ... 
x_bin_cheat_test = np.hstack((x_bin_test,cheatcol_test))
```
The SVM-model is, in principle, able to construct a 100% accurate classifier for this data: we only have to 'learn' that only the last column in the data matters.

f) Describe, briefly, how the coefficients and weights of an SVM model would have to be set, so that the resulting model is 100% accurate on this cheating data. This part of the exercise does not involve any code. Just give your answer in a short text.

g) Investigate how the accuracy of different SVM classifiers improves in practice on this new dataset. Do you achieve 100% accuracy on the test set? If not, try to change the encoding in the cheat column: instead of representing digit1 with a 1, use a larger number, e.g. 250. Does that help? Why?

Exercise 4: Data exploration and logistic regression
In this exercise, you are going to investigate student dropout based on the dataset "churn.cvs". This is a real dataset, and there is no single "correct" way to use it (however, there are several wrong ones!). Your exercise is to explore one or more possible usecases, and document the one(s) you find the most fruitful/interesting. Your work should probably include the steps below:

An investigation of the data, using e.g. FACETs, Pandas, and/or whatever other tools you prefer. Can you find any interesting correlations? Are there problematic features or rows in the dataset?
Handle missing data and possible outliers (in each case, consider what you want to do: Remove row? Remove column? Insert custom value?).
Normalize/bin/create dummy variables where relevant.
Determine what you would like to predict, i.e. choose your target variable. Try formulating a specific usecase for your experiment (e.g. "Given a students perfomance in high school and first semester, what is the probability that he/she churns in the 2. semester?")
Train a logistic regression and at least one other algorithm on the data. Use either manual tuning or cross validation to find a good set of hyperparameters for each model. Do you see any specific advantages in using a logistic regression in this case?
What features seem to be important for predicting whether a student is likely to drop out?
Warning: Make sure you carefully consider what information is available at the time where a prediction is to be made - for example, it doesn't make any sense to try to predict if a student churns in semester 1, if you include a feature which tells that this student churned in semester 2! So depending on your specific usecase, you should probably remove some columns and/or rows before you train your model.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

print("\nSize of Images in digits dataset\t" + str(digits.images.shape)+"\n")
#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


best_acc = -1.0
best_model = None
best_h_params = None

print("\nGamma, C\t","\t\t", "Train\t", "Dev\t", "Test\t")

# 2. For every combination-of-hyper-parameter values
for cur_h_params in h_param_comb:

    #PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #PART: setting up hyperparameter
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)


    #PART: Train model
    # 2.a train the model 
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # print(cur_h_params)
    #PART: get dev set predictions
    predicted_dev = clf.predict(X_dev)
    predicted_train = clf.predict(X_train)
    predicted_test = clf.predict(X_test)

    # 2.b compute the accuracy on the validation set
    cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
    cur_acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
    cur_acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
    

    print(cur_h_params, "\t", round(cur_acc_train, 3), "\t", round(cur_acc, 3), "\t", round(cur_acc_test, 3))
    

    # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_model = clf
        best_h_params = cur_h_params
        # print("Found new best acc with :")
        # print(cur_h_params, "\t", round(cur_acc_train, 3), "\t", round(cur_acc, 3), "\t", round(cur_acc_test, 3))
        # print("Found new best acc with :"+str(cur_h_params))
        # print("New best val accuracy:" + str(cur_acc))



    
#PART: Get test set predictions
# Predict the value of the digit on the test subset

clf.set_params(**best_h_params)

predicted_test = best_model.predict(X_test)
predicted_train = clf.predict(X_train)
predicted_dev = clf.predict(X_dev)

acc_train = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
acc_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
acc_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)


print("Best hyperparameters were:")
print(cur_h_params,"\t", round(acc_train,3) ,"\t",round(acc_dev,3) ,"\t",round(acc_test,3) )



def print_accuracy(accuracy):
    print('Minimum Accuracy {}'.format(np.min(accuracy)))
    print('Max Accuracy {}'.format(np.min(accuracy)))
    print('Mean Accuracy {}'.format(np.mean(accuracy)))
    print('Median Accuracy {}'.format(np.median(accuracy)))
    print('\n')

print('train accuracy')
print_accuracy(acc_train)
print('dev accuracy')
print_accuracy(acc_dev)
print('test accuracy')
print_accuracy(acc_test)


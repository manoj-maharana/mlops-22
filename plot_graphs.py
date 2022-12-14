# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
import argparse
from sklearn import datasets, svm, metrics, tree
import pdb

from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# parser = argparse.ArgumentParser()
# # Adding Argument
# parser.add_argument('--x', type=int,
#                     required=True)
# parser.add_argument('--y', type=str,
#                     required=True)
# random_states = parser.parse_args()
# print(random_states.x)
# print(random_states.y)
# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score

n_cv = 5
results = {}

def main_method(clf_names,random_state):
    for n in range(n_cv):
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac,random_state
        )
        # PART: Define the model
        # Create a classifier: a support vector classifier
        models_of_choice={}
        if clf_names == 'svm':
            models_of_choice['svm']=svm.SVC()
        else:
            models_of_choice['decision_tree']=tree.DecisionTreeClassifier()
        

   
        for clf_name in models_of_choice:
            clf = models_of_choice[clf_name]
            print("[{}] Running hyper param tuning for {}".format(n,clf_name))
            actual_model_path = tune_and_save(
                clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
            )

            # 2. load the best_model
            best_model = load(actual_model_path)
            print(best_model)

            # PART: Get test set predictions
            # Predict the value of the digit on the test subset
            predicted = best_model.predict(x_test)
            if not clf_name in results:
                results[clf_name]=[]    

            results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
            # 4. report the test set accurancy with that best model.
            # PART: Compute evaluation metrics
            print(
                f"Classification report for classifier {clf}:\n"
                f"{metrics.classification_report(y_test, predicted)}\n"
            )
            model_name ="model saved at:"+ actual_model_path
            print(model_name)
            txt_file_name = "results/"+ str(clf_names) + "_" + str(random_state)+ ".txt"
            print(txt_file_name)
            with open(txt_file_name, "w") as text_file:
                text_file.write(str(model_name) + str(results))


# print(results)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_names")
    parser.add_argument("--random_state")
    args = parser.parse_args()
    main_method(clf_names = str(args.clf_names), random_state = int(args.random_state))
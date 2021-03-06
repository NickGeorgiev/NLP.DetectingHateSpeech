import json
import nltk
from nltk import MaxentClassifier
from nltk.classify import NaiveBayesClassifier, SklearnClassifier
from nltk.metrics import precision, recall, f_measure
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from collections import defaultdict

from extract_features import extract_features_of_tweet

TEST_SET_SIZE = 0.2


featuresets = []
with open('../jsons/features_sets.json', 'r') as file:
    featuresets = json.load(file)


data_size = len(featuresets)
train_set_end = round(data_size * (1 - TEST_SET_SIZE))
train_set, test_set = featuresets[:train_set_end], featuresets[train_set_end:]


# naive_bayes = NaiveBayesClassifier.train(train_set)
# print("Accuracy - Naive Bayes Classifier: ")
# print(nltk.classify.accuracy(naive_bayes, test_set))
# print("Most informative features - Naive Bayes Classifier:")
# print(naive_bayes.show_most_informative_features())

# maxent = MaxentClassifier.train(train_set, 'GIS', trace=0,
#                                 encoding=None, gaussian_prior_sigma=0, max_iter=100)
# print("Accuracy - Max Entropy Classifier: ")
# print(nltk.classify.accuracy(maxent, test_set))
# print("Most informative features - Max Entropy Classifier:")
# print(maxent.show_most_informative_features())

linear_svm_classifier = nltk.SklearnClassifier(LinearSVC(C=2.0, dual=True, fit_intercept=True,
                                                         intercept_scaling=0.1, loss='squared_hinge',
                                                         max_iter=1500, penalty='l2', random_state=0,
                                                         tol=0.0001), sparse=False)
linear_svm_classifier.train(train_set)
print("Accuracy - Linear SVM Classifier: ")
print(nltk.classify.accuracy(linear_svm_classifier, test_set))


nonlinear_svm = SklearnClassifier(SVC(gamma='scale', kernel='poly', coef0 = 5.0, degree = 5, C = 5.0, shrinking=True, probability=False, tol=1e-3), sparse=False).train(train_set)
print("Accuracy - Nonlinear SVM: ")
print(nltk.classify.accuracy(nonlinear_svm, test_set))


random_forest = SklearnClassifier(RandomForestClassifier(n_estimators = 100,
                                                         criterion = 'gini',
                                                         max_depth = 5,
                                                         min_samples_split = 2,
                                                         min_samples_leaf = 1,
                                                         min_weight_fraction_leaf = 0.0,
                                                         max_features = 25,
                                                         max_leaf_nodes = 20,
                                                         min_impurity_decrease = 0.0,
                                                         bootstrap = True,
                                                         oob_score = False,
                                                         random_state = None ),
                                  sparse = False)
random_forest.train(train_set)
print("Accuracy - Random Forest Classifier: ")
print(nltk.classify.accuracy(random_forest, test_set))


test_tweet = "75% of illegal Aliens commit Felons such as ID, SSN and Welfare Theft Illegal #Immigration is not a Victimless Crime !"
# print(naive_bayes.classify(extract_features_of_tweet(test_tweet, raw=True)))
# print(maxent.classify(extract_features_of_tweet(test_tweet, raw=True)))
print(linear_svm_classifier.classify(extract_features_of_tweet(test_tweet, raw=False)))
print(nonlinear_svm.classify(extract_features_of_tweet(test_tweet, raw=True)))

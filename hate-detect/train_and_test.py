import json
import nltk
from nltk import MaxentClassifier
from nltk.classify import NaiveBayesClassifier, SklearnClassifier
from sklearn.svm import LinearSVC, SVC

from extract_features import extract_features_of_tweet

TEST_SET_SIZE = 0.2


featuresets = []
with open('../jsons/features_sets.json', 'r') as file:
    featuresets = json.load(file)

data_size = len(featuresets)
train_set_end = round(data_size * (1 - TEST_SET_SIZE))
train_set, test_set = featuresets[:train_set_end], featuresets[train_set_end:]

naive_bayes = NaiveBayesClassifier.train(train_set)
print("Accuracy - Naive Bayes Classifier: ")
print(nltk.classify.accuracy(naive_bayes, test_set))
print("Most informative features - Naive Bayes Classifier:")
print(naive_bayes.show_most_informative_features())

maxent = MaxentClassifier.train(train_set, 'GIS', trace=0,
                                encoding=None, gaussian_prior_sigma=0, max_iter=300)
print("Accuracy - Max Entropy Classifier: ")
print(nltk.classify.accuracy(maxent, test_set))
print("Most informative features - Max Entropy Classifier:")
print(maxent.show_most_informative_features())

linear_svm_classifier = nltk.SklearnClassifier(LinearSVC(C=1.0, dual=True, fit_intercept=True,
                                                         intercept_scaling=0.1, loss='squared_hinge',
                                                         max_iter=600, penalty='l2', random_state=0,
                                                         tol=0.0001), sparse=False)
linear_svm_classifier.train(featuresets)
print("Accuracy - Linear SVM Classifier: ")
print(nltk.classify.accuracy(linear_svm_classifier, test_set))


nonlinear_svm = SklearnClassifier(SVC(), sparse=False).train(train_set)
print("Accuracy - Nonlinear SVM: ")
print(nltk.classify.accuracy(nonlinear_svm, test_set))


test_tweet = "#The CBP officers arrested the smuggler, a Mexican national who attempted to drive the drugs across the border. The suspect was a part of the DHS’s trusted traveler program called FAST, that stands for free and secure trade for commercial vehicles. The program started after 9/11. THIS MULE WAS A “trusted traveler”!"
print(naive_bayes.classify(extract_features_of_tweet(test_tweet, raw=True)))
print(maxent.classify(extract_features_of_tweet(test_tweet, raw=True)))
print(linear_svm_classifier.classify(extract_features_of_tweet(test_tweet, raw=True)))
print(nonlinear_svm.classify(extract_features_of_tweet(test_tweet, raw=True)))
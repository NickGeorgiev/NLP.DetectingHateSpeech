import csv
import json
import random

from preprocessor import initial_text_clean_up
from extract_features import extract_features_of_tweet

TRAIN_SET_SIZE = 1000


def load_train_data(csv_file):
    data = []
    for line in csv_file:
        if len(line) > 0:
            tweet = line[0]
            tweet = initial_text_clean_up(tweet)
            if len(tweet.split()) > 3:
                data.append(tweet)
    # remove duplicates
    data = list(set(data))
    return data


if __name__ == '__main__':
    sarcastic_csv_file = csv.reader(open('../data/sarcastic_dataset.csv', 'rU'), delimiter='\n')
    sarcastic_data = load_train_data(sarcastic_csv_file)[:TRAIN_SET_SIZE//2]
    print("Number of sarcastic tweets in training data: {}".format(TRAIN_SET_SIZE//2))

    non_sarcastic_csv_file = csv.reader(open('../data/nonsarcastic_dataset.csv', 'rU'), delimiter='\n')
    non_sarcastic_data = load_train_data(non_sarcastic_csv_file)[:TRAIN_SET_SIZE//2]
    print("Number of nonsarcastic tweets in training data: {}".format(TRAIN_SET_SIZE//2))

    labeled_data = ([(tweet, 'sarcastic') for tweet in sarcastic_data] +
                    [(tweet, 'nonsarcastic') for tweet in non_sarcastic_data])
    random.shuffle(labeled_data)

    with open('labeled_data.json', 'w') as file:
        file.write(json.dumps(labeled_data))
    print("Labeled data was written to 'labeled_data.json'.")

    featuresets = [(extract_features_of_tweet(tweet), cls) for (tweet, cls) in labeled_data]
    with open('features_sets.json', 'w') as file:
        file.write(json.dumps(featuresets))
    print("Feature sets were written to 'features_sets.json'.")

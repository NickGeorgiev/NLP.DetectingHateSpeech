import nltk
import json
import re
from textblob import TextBlob


from preprocessor import *
from utilities import *

from topic_model import extract_topic_features

from word_embeddings import extract_glove_features

COMMON_UNIGRAMS = set()
COMMON_BIGRAMS = set()
COMMON_TRIGRAMS = set()
COMMON_POS_BIGRAMS = set()
COMMON_POS_TRIGRAMS = set()
COMMON_HASHTAGS=set()

BAD_WORDS = []

with open('../jsons/unigrams.json', 'r') as file:
    COMMON_UNIGRAMS = set(json.load(file))
with open('../jsons/bigrams.json', 'r') as file:
    COMMON_BIGRAMS = set(json.load(file))
with open('../jsons/trigrams.json', 'r') as file:
    COMMON_TRIGRAMS = set(json.load(file))

with open('../jsons/pos_bigrams.json', 'r') as file:
    COMMON_POS_BIGRAMS = set(json.load(file))
with open('../jsons/pos_trigrams.json', 'r') as file:
    COMMON_POS_TRIGRAMS = set(json.load(file))

with open('bad-words.txt', 'r') as file:
    BAD_WORDS = [remove_escaped_characters(word) for word in file.readlines()]

with open('../jsons/common_hashtags.json','r') as file:
    COMMON_HASHTAGS = set(json.load(file))


def extract_sentiment_features_of_tweet(features, tweet):
    tweet = get_tweet_text_only(tweet)

    tokens = nltk.word_tokenize(tweet)

    # Extract features of full sentence.
    tweet_blob = TextBlob(' '.join(tokens))
    features['tweet_polarity'] = tweet_blob.sentiment.polarity
    features['tweet_subjectivity'] = tweet_blob.sentiment.subjectivity



def extract_capitalization_features(features, text):
    capitalized_phrases = [t for t in get_capitalized_text(text) if len(t) > 1]
    features['capitalization_presence'] = int(len(capitalized_phrases) > 0)
    polarity = 0.0
    for phrase in capitalized_phrases:
        polarity += TextBlob(phrase).polarity
    if len(capitalized_phrases):
        polarity /= len(capitalized_phrases)
    features['capitalization_polarity'] = polarity


def extract_interjections_features(features, text):
    interjection_words_descriptions = get_interjection_words_descriptions(text)
    polarity = 0.0
    subjectivity = 0.0
    for interj in interjection_words_descriptions:
        blob = TextBlob(interj)
        polarity += blob.polarity
        subjectivity += blob.subjectivity
    if len(interjection_words_descriptions):
        polarity /= len(interjection_words_descriptions)
        subjectivity /= len(interjection_words_descriptions)
    features['interjections_polarity'] = polarity
    features['interjections_subjectivity'] = subjectivity


def extract_hashtag_features(features, text):
    hashtags = get_hashtags(text)
    polarity = 0.0
    subjectivity = 0.0
    for popular_tag in COMMON_HASHTAGS:
        features['popular_hashtag:' + popular_tag] = 0

    for tag in hashtags:
        blob = TextBlob(tag)
        polarity += blob.polarity
        subjectivity += blob.subjectivity
        if tag in COMMON_HASHTAGS:
            features['popular_hashtag:' + tag] = 1

    if len(hashtags):
        polarity /= len(hashtags)
        subjectivity /= len(hashtags)
    features['hashtags_polarity'] = polarity
    features['hashtags_subjectivity'] = subjectivity


def extract_punctuation_features(features, text):
    features["punctuation_feature"] = len(get_punctuation(text))


def extract_quoted_text_features(features, text):
    features["quoted_text"] = len(get_quoted_text(text))


def extract_quoted_text_polarity(features, text):
    quotes = get_quoted_text(text)
    polarity = 0.0
    for _, quote in quotes:
        quote_blob = TextBlob(quote)
        polarity += quote_blob.sentiment.polarity
    features["quoted_text_polarity"] = -polarity


def extract_bad_words_count(features, tweet):
    tweet = get_tweet_text_only(tweet)
    tokens = nltk.word_tokenize(tweet)

    features['bad_words_count'] = len([token for token in tokens if token in BAD_WORDS]) / (len(tokens)+0.00001)


def extract_ngrams_features(features, text):
    text = get_tweet_text_only(text)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]

    bigrams = nltk.bigrams(tokens)
    bigrams = ['{0} {1}'.format(bi[0], bi[1]) for bi in bigrams]
    trigrams = nltk.trigrams(tokens)
    trigrams = ['{0} {1} {2}'.format(tri[0], tri[1], tri[2]) for tri in trigrams]

    features['common_words_count'] = 0

    for token in tokens:
        if token in COMMON_UNIGRAMS:
            features['common_words_count'] += 1

    for bigram in COMMON_BIGRAMS:
        features['bigram: ' + bigram] = 1 if bigram in bigrams else 0

    for trigram in COMMON_TRIGRAMS:
        features['trigram: ' + trigram] = 1 if trigram in trigrams else 0


def extract_pos_ngrams_features(features, text):
    pos_bigrams, pos_trigrams = get_pos_ngrams(text)

    for pos_bigram in COMMON_POS_BIGRAMS:
        features['pos_bigram: ' + pos_bigram] = 1 if pos_bigram in pos_bigrams else 0

    for pos_trigram in COMMON_POS_TRIGRAMS:
        features['pos_trigram: ' + pos_trigram] = 1 if pos_trigram in pos_trigrams else 0


#looks up whether there are pronouns that should be considered as othering language in the whole sentence
def extract_othering_language_features(features, text):
    text = get_pos_sentence(text)
    adj_prp_text = [[word,tag] for word, tag in text if tag in ['PRP', 'PRP$', 'JJ']]

    #filter founded pronouns and adjectives to exclude ones that should be considered as outgroup language
    filtered_text = [word for word, tag in adj_prp_text if word in outgroup_pronouns or word == outgroup_adjective]
    features['outgroup_language_coef'] =  len(filtered_text) / (len(adj_prp_text) + 0.00001)


def extract_othering_language_collocations(features, text):
    text = get_pos_sentence(text)
    allText = ''.join([tag for word,tag in text])
    vrb_prn_tuples = re.findall(verb_pronoun_regex, allText)
    features['othering_tuples_polarity'] = 0
    polarity = 0
    for verb, pronoun in vrb_prn_tuples:
        if pronoun == outgroup_adjective or pronoun in outgroup_pronouns:
            polarity += TextBlob(verb).sentiment.polarity
    if len(vrb_prn_tuples):
        polarity /= len(vrb_prn_tuples)
    features['othering_tuples_polarity'] = polarity


def count_adjectives(features, text):
    text = get_pos_sentence(text)
    features['adjectives_count'] = len([word for word, tag in text if tag in ['JJ', 'JJR', 'JJS']]) / (len(text) + 0.00001)


def extract_features_of_tweet(tweet, raw=False):
    features = {}
    if raw == False:
        tweet = initial_text_clean_up(tweet)
    tweet = remove_unicode_characters(tweet)
    tweet = remove_escaped_characters(tweet)
    #extract_glove_features(features, tweet)
    extract_punctuation_features(features, tweet)
    extract_quoted_text_features(features, tweet)
    extract_capitalization_features(features, tweet)
    extract_quoted_text_polarity(features, tweet)
    extract_hashtag_features(features, tweet)
    extract_bad_words_count(features, tweet)
    extract_othering_language_features(features,tweet)
    count_adjectives(features,tweet)
    extract_interjections_features(features, tweet)
    extract_ngrams_features(features, tweet)
    extract_pos_ngrams_features(features, tweet)
    extract_punctuation_features(features, tweet)
    #extract_othering_language_collocations(features, tweet)
    extract_sentiment_features_of_tweet(features, tweet)
    extract_topic_features(features, tweet)

    return features

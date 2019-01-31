import nltk
import json
from textblob import TextBlob


from preprocessor import *
from utilities import *


COMMON_UNIGRAMS = set()
COMMON_BIGRAMS = set()
COMMON_TRIGRAMS = set()
COMMON_POS_BIGRAMS = set()
COMMON_POS_TRIGRAMS = set()

BAD_WORDS = []

# with open('unigrams.json', 'r') as file:
#     COMMON_UNIGRAMS = set(json.load(file))
# with open('bigrams.json', 'r') as file:
#     COMMON_BIGRAMS = set(json.load(file))
# with open('trigrams.json', 'r') as file:
#     COMMON_TRIGRAMS = set(json.load(file))

# with open('pos_bigrams.json', 'r') as file:
#     COMMON_POS_BIGRAMS = set(json.load(file))
# with open('pos_trigrams.json', 'r') as file:
#     COMMON_POS_TRIGRAMS = set(json.load(file))

with open('bad-words.txt', 'r') as file:
    BAD_WORDS = [remove_escaped_characters(word) for word in file.readlines()]


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
    for tag in hashtags:
        blob = TextBlob(tag)
        polarity += blob.polarity
        subjectivity += blob.subjectivity
    if len(hashtags):
        polarity /= len(hashtags)
        subjectivity /= len(hashtags)
    features['hashtags_polarity'] = polarity
    features['hashtags_subjectivity'] = subjectivity


# def extract_parsed_sentence_features(features, text):
#     # additional clean up just so that parsing can be done safe
#     text = remove_hashtags(text)
#     text = remove_hashtag_symbol(text)
#     text = text.replace(')', '')
#     text = text.replace('(', '')

#     subclauses = get_subclauses(text)

#     max_polarity = 0.0
#     min_polarity = 0.0
#     for subclause in subclauses['S']:
#         blob = TextBlob(subclause)
#         if blob.polarity > max_polarity:
#             max_polarity = blob.polarity
#         elif blob.polarity < min_polarity:
#             min_polarity = blob.polarity
#     features["subclauses_opposing_polarities"] = abs(max_polarity - min_polarity)


def extract_punctuation_features(features, text):
    features["punctuation_feature"] = len(get_punctuation(text))


def extract_quoted_text_features(features, text):
    features["quoted_text"] = len(get_quoted_text(text))


def extract_quoted_text_polarity(features, text):
    quotes = get_quoted_text(text)
    polarity = 0.0
    for startPos, quote in quotes: #sth with the regex?
        quote_blob = TextBlob(quote)
        polarity += quote_blob.sentiment.polarity
    features["quoted_text_polarity"] = -polarity


def extract_bad_words_count(features, tweet):
    tweet = get_tweet_text_only(tweet)
    tokens = nltk.word_tokenize(tweet)

    features['bad_words_count'] = len([token for token in tokens if token in BAD_WORDS])


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


def extract_features_of_tweet(tweet, raw = False):
    features = {}
    if raw == False:
        tweet = initial_text_clean_up(tweet)
    tweet = remove_unicode_characters(tweet)
    tweet = remove_escaped_characters(tweet)
    extract_punctuation_features(features, tweet)
    extract_quoted_text_features(features, tweet)
    extract_capitalization_features(features, tweet)
    extract_quoted_text_polarity(features, tweet)
    extract_hashtag_features(features, tweet)
    extract_bad_words_count(features, tweet)
    extract_interjections_features(features, tweet)
    extract_ngrams_features(features, tweet)
    extract_pos_ngrams_features(features, tweet)
    extract_punctuation_features(features, tweet)
    #extract_parsed_sentence_features(features, tweet)
    extract_sentiment_features_of_tweet(features, tweet)

    return features



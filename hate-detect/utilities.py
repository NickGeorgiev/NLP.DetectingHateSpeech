import os
import nltk
import string
from nltk.parse import stanford

#os.environ['CLASSPATH'] = '/home/chris/stanford-parser-full-2017-06-09' !!!!!


parse_keys = {'S', 'NP', 'VP'}


def get_subclauses(text):
    parser = stanford.StanfordParser()
    clauses = dict.fromkeys(list(parse_keys), [])
    try:
        trees = [tree for tree in parser.parse(text.split())]
        tree = nltk.Tree.fromstring(str(trees[0]))
        for subtree in tree.subtrees():
            if subtree.label() in parse_keys:
                clauses[subtree.label()].append(' '.join(subtree.leaves()))
        for item in parse_keys:
            clauses[item] = clauses[item][1:]
            clauses[item] = [p for p in clauses[item] if len(p.split()) > 1]
        return clauses
    except Exception as e:
        print(e)
        print("From sentence: {0}".format(text))
    return clauses


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token in string.punctuation or token.isnumeric():
            tokens.remove(token)
    return tokens


def get_pos_ngrams(text):
    tokenized_text = tokenize_text(text)
    pos_tags = nltk.pos_tag(tokenized_text, tagset="universal")
    pos_tag_bigrams = ['{0} {1}'.format(bigram[0][1], bigram[1][1]) for bigram in nltk.bigrams(pos_tags)]
    pos_tag_trigrams = ['{0} {1} {2}'.format(trigram[0][1], trigram[1][1], trigram[2][1]) for trigram in nltk.trigrams(pos_tags)]
    return pos_tag_bigrams, pos_tag_trigrams


def get_pos_sentence(text):
    text = text.lower()
    tokenized_text = tokenize_text(text)
    pos_tagged_text = nltk.pos_tag(tokenized_text)
    return pos_tagged_text


def join_tokens(tokens):
    new_str = ''
    for t in tokens:
        if t not in string.punctuation and not t.isnumeric():
            t = t.strip('"')
            t = t.strip("'")
            new_str += (' ' + t)
    return new_str.strip()

url_regex = r'(ftp|http(s)?)://.*?(?=\s|$)'
hashtag_symbol = '#'
hashtag_regex = r"#(\w+)"
user_mention_regex = r'@\w+\s?'
capitalized_text_regex = r'\b[A-Z\s]+\b'
feature_punctuation = r'\!|\?|\.{2,}'
quotes = r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)"


contractions = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\byou're\b": "you are",
    r"\bwe're\b": "we are",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcnt\b": "can not",
    r"\bcannot\b": "can not",
    r"\bm\b": "am",
    r"\bIm\b": "I am",
    r"\byoure\b": "you are",
    r"\bdont\b": "do not",
    r"\bdoesnt\b": "does not",
    r"\bdidnt\b": "did not",
    r"\bhasnt\b": "has not",
    r"\bhavent\b": "have not",
    r"\bhadnt\b": "had not",
    r"\bwouldnt\b": "would not",
    r"\bcant\b": "can not",
    r"\btheyre\b": "they are",
    r"\bthey're\b": "they are",
    r"\bll\b": "will",
}

interjection_words = {
    'ack': 'disgust dismissal',
    'ah': 'relief or delight',
    'aha': 'triumph surprise',
    'ahem': 'attention',
    'argh': 'frustration',
    'aye': 'agreement',
    'bah': 'dismissal',
    'blah': 'boredom disappointment',
    'blech': 'nausea',
    'bleah': 'nausea',
    'bleh': 'nausea',
    'Boo':  'fright',
    'boo-hoo': 'derisive',
    'boo-ya': 'triumph',
    'bwah-hah-hah': 'triumphant laugh',
    'mwah-hah-hah': 'triumphant laugh',
    'duh': 'mocking',
    'eek': 'unpleasant surprise',
    'eh': 'dismissal',
    'ew': 'disgust',
    'feh': 'underwhelmed disappointed',
    'gak': 'disgust distaste',
    'ha': 'joy surprise triumph',
    'ha-ha': 'laughter',
    'hee-hee': 'mischievous laugh',
    'hee': 'mischievous laugh',
    'heh-heh': 'derisive laugh',
    'hm': 'skepticism',
    'hmph': 'displeasure indignation',
    'ho-ho': 'mirth',
    'oh-ho': 'triumph of discovery',
    'ho-hum': 'indifference or boredom',
    'hubba-hubba': 'leer',
    'huh': 'disbelief confusion surprise',
    'hurrah': 'triumph happiness',
    'hoorah': 'triumph happiness',
    'hooray': 'triumph happiness',
    'hurray': 'triumph happiness',
    'ick': 'disgust',
    'lah-de-dah': 'nonchalance dismissal',
    'mm-hmm': 'affirmative or corroborating response',
    'mmm': 'palatable or palpable pleasure',
    'mwah': 'kiss affection',
    'oh-oh': 'worry',
    'olé': 'celebrate deft or adroit maneuver',
    'ooh': 'interest or admiration',
    'oops': 'error fault',
    'oopsie': 'error fault',
    'oopsy': 'error fault',
    'whops': 'error fault',
    'ouch': 'pain',
    'ow' : 'pain',
    'oy': 'frustration concern self-pity',
    'pff': 'disappointment disdain annoyance',
    'pfft': 'disappointment disdain annoyance',
    'phfft': 'disappointment disdain annoyance',
    'phew': 'relief',
    'pooh': 'contempt',
    'tsk-tsk': 'condemnation scolding',
    'tch': 'disapproval',
    'ugh': 'disgust',
    'uh': 'skepticism',
    'uh-huh': 'affirmation agreement',
    'uh-oh': 'concern dismay',
    'uh-uh': 'negation refusal',
    'um': 'skepticism',
    'whee': 'excitement delight',
    'whew': 'relief',
    'whoa': 'surprise',
    'whoop-de-doo': 'mockery',
    'woo': 'excitement',
    'woo-hoo': 'excitement',
    'woot': 'excitement',
    'yahoo': 'excitement',
    'yee-haw': 'excitement',
    'yippee': 'excitement',
    'wow': 'surprise amazement',
    'yay':  'hapiness excitement',
    'yaay': 'happiness excitement',
    'yaaay': 'happiness excitement',
    'yikes': 'concern',
    'yow': 'surprised impressed',
    'yowza': 'impressed surpised',
    'yuck': 'disgust',
    'yum': 'delicious good',
    'yummy': 'delicious good'
}

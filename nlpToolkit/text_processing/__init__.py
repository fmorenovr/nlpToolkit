from .text_metrics import TextMetric
from .text_explainer import TextExplainer
from .text_processing import TextProcesser

# import nltk
# str_example = "Pythons live near the equator, ***IMG***[https://i.imgur.com/WEy0JkM.png]*** in Asia and Africa, where it is hot and wet and their huge bodies can stay warm.  They make their homes in caves or in trees and have become used to living in cities and towns since people have been moving in on their territory."
# tokens = nltk.word_tokenize(str_example)
# print('Tokens')
# print(tokens)

# from nltk.stem.snowball import SnowballStemmer

# stemmer = SnowballStemmer(language='english')
# stemd_word = [stemmer.stem(plural) for plural in tokens]
# print('Stemmed Form')
# print(stemd_word)

# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()
# lemma_word = [lemmatizer.lemmatize(plural) for plural in tokens]
# print('Lemma Form')
# print(lemma_word)

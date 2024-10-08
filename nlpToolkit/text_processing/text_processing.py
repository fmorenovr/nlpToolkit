#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from tqdm.auto import tqdm
import re
import spacy
import copy
import itertools

from .text_filter import TextFilter

class TextProcesser:
    """Full text preprocessing with spaCy."""
    def __init__(self, language_to_process=None, 
                       to_lemma=True, 
                       to_stem=False, 
                       to_filter_char=False,
                       word_min_len=0,
                       keep_stopwords=False, 
                       keep_puncts=False, 
                       keep_specials=False, 
                       keep_singles=False,
                       keep_emojis=False,
                       keep_currency=False,
                       keep_digits=False,
                       keep_spaces=False,
                       keyword_sep="FAMVEER",
                       log_tqdm=True, 
                       exclude_pipe = ['morphologizer', "parser", "tagger", 'attribute_ruler', "ner"],
                       n_jobs=6):
        """WordCloud Drawer.
        Args:
            language to process (str): language to be use as nlp.
            to_lemma (bool): True to lemmatization. default True
            to_stem (bool): True to stemming. default False.
            keep_stops (bool): keep language stop words, default False.
            keep_puncts (bool): keep language punctuations, default False.
            log_tqdm (bool): True to print tqdm progress bar.
            n_jobs (int): number of processor to be used.
        """
        
        self.language_to_process = language_to_process
        self.exclude_pipe = exclude_pipe
        self.n_jobs = n_jobs
        self.to_lemma = to_lemma
        self.to_stem = to_stem
        
        self.word_min_len = word_min_len
        self.keep_stopwords = keep_stopwords
        self.keep_puncts = keep_puncts
        self.keep_specials = keep_specials
        self.keep_singles = keep_singles
        self.keep_emojis = keep_emojis
        self.keep_currency = keep_currency
        self.keep_digits = keep_digits
        self.keep_spaces = keep_spaces
        
        self.keyword_sep = keyword_sep.lower()
        self.log_tqdm = log_tqdm
        self.txt_filter = TextFilter(None, 
                                      keep_stopwords=keep_stopwords, 
                                      keep_puncts=keep_puncts, 
                                      keep_specials=keep_specials,
                                      keep_singles=keep_singles,
                                      keep_emojis=keep_emojis,
                                      keep_currency=keep_currency, 
                                      keep_digits=keep_digits, 
                                      keep_spaces=keep_spaces,
                                      keyword_sep=self.keyword_sep)
        
        if self.language_to_process is not None:
            self.create_nlp()
        
        if exclude_pipe is not None:
            self.set_nlp_pipeline(exclude_pipe)

    def set_nlp_language(self, language_to_process):
        self.language_to_process = language_to_process
        if self.to_stem:
            self.stemmer = SnowballStemmer(language=self.language_to_process)
        self.create_nlp()

    def get_nlp_language(self):
        return self.nlp
    
    def set_nlp_pipeline(self, pipe):
        self.exclude_pipe = pipe

    def get_current_language(self):
        return self.language_to_process
    
    def get_nlp_pipeline(self):
        return self.nlp.pipeline, self.nlp.pipe_names
    
    def get_nlp_stopwords(self):
        return list(self.nlp.Defaults.stop_words)
    
    def get_nlp_punctuations(self):
        from string import punctuation
        punct_ = set(set(self.nlp.Defaults.prefixes)).union(set(punctuation))
        return list(punct_)

    def create_nlp(self):
        """Create spaCy's NLP object.
        We disable almost all of the default spaCy pipeline components to
        speed up the text processing.
        Documentation: https://spacy.io/models
        
        NLP default has this pipeline:
        tok2vec: Convert raw text into word vectors (also known as word embeddings) that represent the meaning of each word in a document.
        morphologizer: handles morphological analysis, providing information about the word's lemma, part-of-speech (POS) tag, and other morphological features.
                       also, enable the generation of word forms from lemmas and morphological specifications.
        tagger: Part-of-Speech Tag Word Class (e.g. many ADJECTIVE, 2.33 NUMBER, etc.)
        parser: Grammatical Morphological Analyzer building a dependency tree (e.g. the DETERMINANT, has AUXILIAR, in PREPOSITION, etc.)
        lemmatizer: handles lemmatization, which is the process of reducing words to their base or canonical form (lemmas).
        attribute_ruler: Maps tagger and parser
        ner: Named Entity Recognition (e.g. matheus is PERSON, 2.33 is CARDINAL, yesterday is DATE, etc.)
        """
        if self.language_to_process=="english" or self.language_to_process=="en":
            try:
                import en_core_web_sm
            except:
                spacy.cli.download("en_core_web_sm") # English
                import en_core_web_sm
            self.nlp = en_core_web_sm.load(exclude=self.exclude_pipe) if len(self.exclude_pipe)>0 else spacy.load('en_core_web_sm')
        elif self.language_to_process=="russian" or self.language_to_process=="ru":
            try:
                import ru_core_news_sm
            except:
                spacy.cli.download("ru_core_news_sm") # Russian
                import ru_core_news_sm
            self.nlp = ru_core_news_sm.load(exclude=self.exclude_pipe) if len(self.exclude_pipe)>0 else spacy.load('ru_core_news_sm')
        elif self.language_to_process=="portuguese" or self.language_to_process=="pt":
            try:
                import pt_core_news_sm
            except:
                spacy.cli.download("pt_core_news_sm") # Portuguese
                import pt_core_news_sm
            self.nlp = pt_core_news_sm.load(exclude=self.exclude_pipe) if len(self.exclude_pipe)>0 else spacy.load('pt_core_news_sm')
        elif self.language_to_process=="deutsch" or self.language_to_process=="de":
            try:
                import de_core_news_sm
            except:
                spacy.cli.download("de_core_news_sm") # Deutsch
                import de_core_news_sm
            self.nlp = de_core_news_sm.load(exclude=self.exclude_pipe) if len(self.exclude_pipe)>0 else spacy.load('de_core_news_sm')
        elif self.language_to_process=="spanish" or self.language_to_process=="es":
            try:
                import es_core_news_sm
            except:
                spacy.cli.download("es_core_news_sm") # Spanish
                import es_core_news_sm
            self.nlp = es_core_news_sm.load(exclude=self.exclude_pipe) if len(self.exclude_pipe)>0 else spacy.load('es_core_news_sm')
        else:
            try:
                import en_core_web_sm
            except:
                spacy.cli.download("en_core_web_sm") # English
                import en_core_web_sm
            self.nlp = en_core_web_sm.load(exclude=self.exclude_pipe) if len(self.exclude_pipe)>0 else spacy.load('en_core_web_sm')

        from spacy.tokenizer import Tokenizer
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        
        self.nlp.add_pipe("text_filter", after="tok2vec", config={"keep_stopwords": self.keep_stopwords, 
                                                                  "keep_puncts": self.keep_puncts, 
                                                                  "keep_specials": self.keep_specials,  
                                                                  "keep_singles": self.keep_singles, 
                                                                  "keep_emojis": self.keep_emojis, 
                                                                  "keep_currency": self.keep_currency,
                                                                  "keep_digits": self.keep_digits,
                                                                  "keep_spaces": self.keep_spaces,
                                                                   })
        self.nlp.max_length = 10000000

    def generate_tokens_from_text(self, doc):
    
        filtered_tokens = [token for token in doc]
        
        if not self.keep_stopwords:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_stopword]
        
        if not self.keep_puncts:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_punctuation]
        
        if not self.keep_specials:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_special_character]
        
        if not self.keep_emojis:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_emoji]
        
        if not self.keep_currency:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_currency]
        
        if not self.keep_digits:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_digit]
            
        if not self.keep_singles:
            filtered_tokens = [token for token in filtered_tokens if not token._.is_single_character]
    
        #filtered_tokens = [token for token in doc if not token._.to_remove]
    
        if self.to_lemma or self.to_stem:
            # Tokenization, lemmatization or stemming, and convertion to lowercase
            token_list = [token.lemma_.lower() if self.to_lemma else self.stemmer.stem(token.text.lower()) for token in filtered_tokens]
        else:
            # Tokenization and convertion to lowercase
            token_list = [token.text.lower() for token in filtered_tokens]
        
        token_list = [token.strip() for token in token_list]
        token_list = [token.strip() for token in token_list if len(token)>self.word_min_len]
        token_list = [token.strip() for token in token_list if token!=self.keyword_sep]
        
        return token_list

    def process_texts_by_language(self, texts):
        """Full text preprocessing.
        Args:
            texts (list of str): List of texts to be preprocessed.
        Returns:
            list of str: List of preprocessed texts.
        """
        if type(texts)==str:
            texts = [texts]

        processed_tokens = []

        if self.log_tqdm:
            for doc in tqdm(
                self.nlp.pipe(texts, n_process=self.n_jobs),
                desc="Preprocessing texts",
                total=len(texts)
            ):
                final_tokens = self.generate_tokens_from_text(doc)

                processed_tokens.append(" ".join(final_tokens))
        else:
            for doc in self.nlp.pipe(texts, n_process=self.n_jobs):
                final_tokens = self.generate_tokens_from_text(doc)

                processed_tokens.append(" ".join(final_tokens))

        self.processed_texts = processed_tokens

        return processed_tokens
    
    def get_processed_texts_by_language(self):
        return self.processed_texts
    
    def get_wordlist_from_processed_texts_by_language(self):
        texts_tokens = [txt.split(" ") for txt in self.processed_texts]

        return list(itertools.chain(*texts_tokens))
    
    def process_wordlist(self, tokens, languages_list=[]):

        if type(languages_list)==str:
            languages_list = [languages_list]

        filter_words = []
        punct_chars = []
        for lang in languages_list:
            self.set_nlp_language(language_to_process=lang)
            stop_words_ = self.get_nlp_stopwords()
            punctuations_ = self.get_nlp_punctuations()
            
            filter_words.append(stop_words_)
            punct_chars.append(punctuations_)
        
        filter_words = list(itertools.chain(*filter_words))
        filter_puncts = list(set(itertools.chain(*punct_chars)))
        
        word_list = [ token for token in tokens if token not in filter_words and token not in filter_puncts]
        
        word_list = self.txt_filter.filter(word_list)
        self.word_list = word_list
        
        return word_list

    def get_processed_wordlist(self, texts):
        return self.word_list
    

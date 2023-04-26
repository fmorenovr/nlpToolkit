#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from tqdm.auto import tqdm
import re
import copy
import itertools

from .text_filter import TextFilter

class TextProcesser:
    """Full text preprocessing with spaCy."""
    def __init__(self, language_to_process=None, 
                       to_lemma=True, 
                       to_stem=False, 
                       keep_stopwords=False, 
                       keep_puncts=False, 
                       keep_specials=False, 
                       keep_emojis=False,
                       keyword_sep="FAMVEER",
                       log_tqdm=True, 
                       exclude_pipe = ["parser", "ner"],
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
        self.n_jobs = n_jobs
        self.to_lemma = to_lemma
        self.to_stem = to_stem
        self.keep_stopwords = keep_stopwords
        self.keep_puncts = keep_puncts
        self.keep_specials = keep_specials
        self.keep_emojis = keep_emojis
        self.keyword_sep = keyword_sep
        self.log_tqdm = log_tqdm
        self.exclude_pipe = exclude_pipe
        self.txt_filter = TextFilter(None, keep_stopwords=True, 
                                      keep_puncts=True, 
                                      keep_currency=True, 
                                      keyword_sep=self.keyword_sep)
        
        if self.language_to_process is not None:
            self.create_nlp()

    def set_nlp_language(self, language_to_process):
        self.language_to_process = language_to_process
        if self.to_stem:
            self.stemmer = SnowballStemmer(language=self.language_to_process)
        self.create_nlp()

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
        tok2vec: Document Embedding
        tagger: Part-of-Speech Tag Word Class (e.g. many ADJECTIVE, 2.33 NUMBER, etc.)
        parser: Grammatical Morphological Analyzer (e.g. the DETERMINANT, has AUXILIAR, in PREPOSITION, etc.)
        attribute_ruler: Maps tagger and parser
        lemmatizer: calculates lemma of each word
        ner: Entity Recognition (e.g. matheus is PERSON, 2.33 is CARDINAL, yesterday is DATE, etc.)
        """
        if self.language_to_process=="english" or self.language_to_process=="en":
            try:
                import en_core_web_sm
            except:
                import spacy
                spacy.cli.download("en_core_web_sm") # English
                import en_core_web_sm
            #import spacy
            #nlp = spacy.load("en_core_web_sm")
            self.nlp = en_core_web_sm.load(exclude=self.exclude_pipe)
        elif self.language_to_process=="russian" or self.language_to_process=="ru":
            try:
                import ru_core_news_sm
            except:
                import spacy
                spacy.cli.download("ru_core_news_sm") # Russian
                import ru_core_news_sm
            #nlp = spacy.load("ru_core_news_sm")
            self.nlp = ru_core_news_sm.load(exclude=self.exclude_pipe)
        elif self.language_to_process=="portuguese" or self.language_to_process=="pt":
            try:
                import pt_core_news_sm
            except:
                import spacy
                spacy.cli.download("pt_core_news_sm") # Portuguese
                import pt_core_news_sm
            #nlp = spacy.load("pt_core_news_sm")
            self.nlp = pt_core_news_sm.load(exclude=self.exclude_pipe)
        elif self.language_to_process=="deutsch" or self.language_to_process=="de":
            try:
                import de_core_news_sm
            except:
                import spacy
                spacy.cli.download("de_core_news_sm") # Deutsch
                import de_core_news_sm
            #nlp = spacy.load("de_core_news_sm")
            self.nlp = de_core_news_sm.load(exclude=self.exclude_pipe)
        else:
            try:
                import en_core_web_sm
            except:
                import spacy
                spacy.cli.download("en_core_web_sm") # English
                import en_core_web_sm
            #nlp = spacy.load("en_core_web_sm")
            self.nlp = en_core_web_sm.load(exclude=self.exclude_pipe)

        from spacy.tokenizer import Tokenizer
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        
        self.nlp.add_pipe("text_filter", after="tok2vec", config={"keep_stopwords": False, 
                                                                   "keep_puncts": False, 
                                                                   "keep_specials": False, 
                                                                   "keep_emojis": False, 
                                                                   "keep_currency": False})
        self.nlp.max_length = 10000000

    def generate_tokens_from_text(self, doc):
    
        filtered_tokens = [token for token in doc if not token._.to_remove]
    
        if self.to_lemma:
            # Tokenization, lemmatization, and convertion to lowercase
            token_list = [token.lemma_.lower() for token in filtered_tokens]
        elif self.to_stem:
            # Tokenization, stemming, and convertion to lowercase
            token_list = [self.stemmer.stem(token.text.lower()) for token in filtered_tokens]
        else:
            # Tokenization and convertion to lowercase
            token_list = [token.text.lower() for token in filtered_tokens]
        
        token_list = [token.strip() for token in token_list]
        token_list = [token.strip() for token in token_list if len(token)>0]
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
    

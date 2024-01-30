#!/usr/bin/python

from ..text_processing import TextProcesser
from nltk.corpus import stopwords

import enchant
import itertools

class LanguageProcesser:
    """Full lang preprocessing with spaCy, Pyenchant, and NLTK."""
    def __init__(self, languages_to_eval=None, 
                       use_spacy=True, 
                       to_lemma=False, 
                       to_stem=False, 
                       word_min_len=0,
                       keep_stopwords=False, 
                       keep_puncts=False, 
                       keep_specials=False, 
                       keep_emojis=False,
                       keep_currency=False,
                       keep_digits=False,
                       keep_spaces=False,
                       exclude_pipe=['morphologizer', "parser", 'attribute_ruler', "ner"],
                       keyword_sep="FAMVEER",
                       log_tqdm=False, 
                       log=False, 
                       nltk_keep_punct=True):
        """Full text preprocessing.
        Args:
            languages_to_eval (list of str): List of languages to eval.
            nltk_keep_punct (bool): Keep punctutation (NLTK package).
        """
        
        self.lang_dict = {"portuguese":"pt_BR",
             "deutsch": "de_DE",
             "german": "de_DE",
             "english": "en_US",
             "russian": "ru_RU",
             "spanish": "es_ES",
            }
        
        self.languages_to_eval = languages_to_eval
        self.use_spacy = use_spacy
        self.to_stem = to_stem
        self.keyword_sep = keyword_sep
        self.log = log
        self.nltk_keep_punct = nltk_keep_punct
        self.txt_processer = TextProcesser(to_lemma=to_lemma, 
                                           to_stem=to_stem,
                                           word_min_len=word_min_len,
                                           keep_stopwords=keep_stopwords, 
                                           keep_puncts=keep_puncts, 
                                           keep_specials=keep_specials, 
                                           keep_emojis=keep_emojis,
                                           keep_currency=keep_currency,
                                           keep_digits=keep_digits, 
                                           keep_spaces=keep_spaces,
                                           keyword_sep=keyword_sep,
                                           exclude_pipe=exclude_pipe,
                                           log_tqdm=log_tqdm, 
                                           )

    # set nlp language
    def set_nlp_language(self, language_to_process):
        self.txt_processer.set_nlp_language(language_to_process=language_to_process)

    def get_nlp_pipeline(self):
        return self.txt_processer.get_nlp_pipeline()

    def spacy_process_texts(self, sentence):
        """Full text preprocessing.
        Args:
            texts (list of str): List of texts to be preprocessed.
            language_to_eval (str): language to eval.
        Returns:
            list of str: List of words.
        """
        txt_spacy = self.txt_processer.process_texts_by_language(sentence)

        split_txt = [txt_.split() for txt_ in txt_spacy]

        words_within_txt = list(set(itertools.chain(*split_txt)))
        total_words_within_txt = list(itertools.chain(*split_txt))

        return words_within_txt, total_words_within_txt, txt_spacy

    # A phrase can be from a word to a sentence
    def get_words_list(self, sentence):
        """Process a list of str.
        Args:
            texts (list of str): List of texts to be preprocessed.
            language_to_eval (str): language to eval.
        Returns:
            list of str: List of words.
        """
        if self.use_spacy:
            words_within_txt, total_words_within_txt, sentence_processed = self.spacy_process_texts(sentence) 
        else:
            words_within_txt, total_words_within_txt, sentence_processed = self.nltk_process_texts(sentence)
        
        return words_within_txt, total_words_within_txt, sentence_processed

    # Evaluate ratios of correct and incorrect words in some language.
    def get_words_ratio(self, word_list, language_to_eval="portuguese"):
        """Process a list of str.
        Args:
            word_list: List of words.
            language_to_eval (str): language to eval.
        Returns:
            relative ratios: words that belong to some language.
            incorrect words: list of incorrect words
            correct words: list of correct words
        """
        lang_d = enchant.Dict(self.lang_dict[language_to_eval])
        total_ = len(word_list)
        relative_ = 0
        incorrect_words = []
        correct_words = []
        
        for word in word_list:
            if len(word)>0:
                if lang_d.check(word):
                    relative_ +=1
                    correct_words.append(word)
                else:
                    incorrect_words.append(word)
        try:
            return relative_/total_, incorrect_words, correct_words
        except: # error divided by 0
            return 0, incorrect_words, correct_words
    
    def get_language_dict(self, sentence):
        """Process a list of str.
        Args:
            sentence (str): text to eval.
        Returns:
            language dict: dict of languages and related info.
        """
        languages_ratios = {}
        
        if self.languages_to_eval is None:
            self.languages_to_eval = stopwords.fileids()
        
        if type(self.languages_to_eval)==str:
            self.languages_to_eval = [self.languages_to_eval]
        
        for language in self.languages_to_eval:
        
            self.set_nlp_language(language)
            if self.log:
                print("Languages", self.get_nlp_pipeline())
        
            languages_ratios[language] = {}

            word_list, total_word_list, text_processed = self.get_words_list(sentence)
            ratio_lang, incorrect_words_lang, correct_words_lang = self.get_words_ratio(word_list, language)
            ratio_lang_all, incorrect_words_lang_all, correct_words_lang_all = self.get_words_ratio(total_word_list, language)
            
            languages_ratios[language]["ratio"] = ratio_lang
            languages_ratios[language]["incorrect_words"] = incorrect_words_lang
            languages_ratios[language]["correct_words"] = correct_words_lang
            languages_ratios[language]["all_incorrect_words"] = incorrect_words_lang_all
            languages_ratios[language]["all_correct_words"] = correct_words_lang_all
            languages_ratios[language]["all_words"] = total_word_list
            languages_ratios[language]["text_processed"] = text_processed
            languages_ratios[language]["ratio_all_words"] = ratio_lang_all
        
        return languages_ratios
    
    # return language ratios, language detected, list of incorrect words, and list of correct words
    def detect_language_and_word_list(self, text):
        lang_freq = self.get_language_dict(text)
        ratios = {}
        ratios_all = {}
        incorrect_words = {}
        all_correct_words = {}
        all_incorrect_words = {}
        correct_words = {}
        proc_txt = {}
        all_words = {}
        
        all_txt_analyzed = []
        
        for lang in self.languages_to_eval:
            ratios[lang] = lang_freq[lang]["ratio"]
            incorrect_words[lang] = lang_freq[lang]["incorrect_words"]
            correct_words[lang] = lang_freq[lang]["correct_words"]
            all_incorrect_words[lang] = lang_freq[lang]["all_incorrect_words"]
            all_correct_words[lang] = lang_freq[lang]["all_correct_words"]
            proc_txt[lang] = lang_freq[lang]["text_processed"]
            all_words[lang] = lang_freq[lang]["all_words"]
            ratios_all[lang] = lang_freq[lang]["ratio_all_words"]
            all_txt_analyzed.append(proc_txt[lang])
        
        if max(ratios.values())>0.0:
            language_detected = max(ratios, key=ratios.get)
            incorrect_words_list = list(incorrect_words[language_detected])
            correct_words_list = list(correct_words[language_detected])
            all_incorrect_words_list = list(all_incorrect_words[language_detected])
            all_correct_words_list = list(all_correct_words[language_detected])
            processed_text = proc_txt[language_detected]
            all_words_ = all_words[lang]
        else:
            language_detected = "not found"
            incorrect_words_list = list(set(itertools.chain(*incorrect_words.values())))
            correct_words_list = list(set(itertools.chain(*correct_words.values())))
            all_incorrect_words_list = list(set(itertools.chain(*all_incorrect_words.values())))
            all_correct_words_list = list(set(itertools.chain(*all_correct_words.values())))
            processed_text = list(set(itertools.chain(*all_txt_analyzed)))
            all_words_ = all_words[lang]

        self.ratios = ratios
        self.ratios_all_words = ratios_all
        self.language_detected = language_detected
        self.incorrect_words = incorrect_words_list
        self.correct_words = correct_words_list
        self.all_incorrect_words = all_incorrect_words_list
        self.all_correct_words = all_correct_words_list
        self.processed_text = processed_text
        self.all_words = all_words_

    def get_processed_text(self):
        return self.processed_text

    # return language detected
    def get_language_ratios(self):
        return self.ratios
    
    # return language detected
    def get_all_language_ratios(self):
        return self.ratios_all_words

    # return ratio of detected language
    def get_detected_language_ratio(self):
        return max(self.ratios.values())

    # return language most rated
    def get_detected_language(self):
        return self.language_detected
        
    def get_all_words(self):
        return self.all_words
        
    def get_all_words_len(self):
        return len(self.all_words)

    # return list of correct words in the most rated language
    def get_correct_words(self):        
        return self.correct_words

    # return length of correct words in the most rated language
    def get_correct_words_len(self):        
        return len(self.correct_words)

    # return list of correct words in the most rated language
    def get_all_correct_words(self):        
        return self.all_correct_words

    # return length of correct words in the most rated language
    def get_all_correct_words_len(self):        
        return len(self.all_correct_words)

    # return list of incorrect words in the most rated language
    def get_incorrect_words(self): 
        return self.incorrect_words

    # return length  of incorrect words in the most rated language
    def get_incorrect_words_len(self):        
        return len(self.incorrect_words)

    # return list of incorrect words in the most rated language
    def get_all_incorrect_words(self): 
        return self.all_incorrect_words

    # return length  of incorrect words in the most rated language
    def get_all_incorrect_words_len(self):        
        return len(self.all_incorrect_words)

    # NLTK procesing texts
    def nltk_process_texts(self, texts, language_to_process=None):
        """Process a list of str.
        Args:
            language_to_process (str): language to eval.
        """
        
        from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
        from nltk.stem.snowball import SnowballStemmer

        from string import punctuation, digits
        
        palavras_txt = []

        if type(texts)==str:
            texts = [texts]
            
        stemmer = SnowballStemmer(language=language_to_process)

        for sentence in texts:

            if language_to_process is not None:
                palavras_inside = word_tokenize(sentence.lower(), language=language_to_process) if self.nltk_keep_punct else wordpunct_tokenize(sentence.lower())
                # avoid numbers (alone) punctuations and stopwords
                stopwords_avoid = set(stopwords.words(language_to_process)).union(set(punctuation)).union(set(digits))

            else:
                palavras_inside = word_tokenize(sentence.lower())
                # Getting StopWords for the language
                stopwords_avoid = set(punctuation).union(set(digits))
            
            palavras_sem_stopwords = [palavra for palavra in palavras_inside if palavra not in stopwords_avoid]

            # for NLTK, we will stemm instead
            if self.to_stem:
                palavras_sem_stopwords = [stemmer.stem(plural) for plural in palavras_sem_stopwords]
            
            palavras_txt.append(palavras_sem_stopwords)
        
        words_within_txt = list(set(itertools.chain(*palavras_txt)))

        return words_within_txt

#!/usr/bin/python

import re
import copy
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

@Language.factory("text_filter", default_config={"keep_stopwords": False, 
                                                 "keep_puncts": False, 
                                                 "keep_specials": False, 
                                                 "keep_emojis": False, 
                                                 "keep_currency": False, 
                                                 "keep_digits": False, 
                                                 "keep_spaces": False,
                                                 })
def create_text_filter(nlp: Language, 
                       name: str, 
                       keep_stopwords: bool, 
                       keep_puncts: bool, 
                       keep_specials: bool, 
                       keep_emojis: bool, 
                       keep_currency: bool,
                       keep_digits: bool, 
                       keep_spaces: bool):
    return TextFilter(nlp, 
                      keep_stopwords=keep_stopwords, 
                      keep_puncts=keep_puncts, 
                      keep_specials=keep_specials, 
                      keep_emojis=keep_emojis, 
                      keep_currency=keep_currency,
                      keep_digits=keep_digits)

class TextFilter:
    def __init__(self, nlp: Language,
                       keep_stopwords=False, 
                       keep_puncts=False, 
                       keep_specials=False, 
                       keep_emojis=False,
                       keep_currency=False,
                       keep_digits=False,
                       keep_spaces=False,
                       keyword_sep="FAMVEER"):
                       
        self.keep_stopwords = keep_stopwords
        self.keep_puncts = keep_puncts
        self.keep_specials = keep_specials
        self.keep_emojis = keep_emojis
        self.keep_currency = keep_currency
        self.keep_digits = keep_digits
        self.keep_spaces = keep_spaces
        self.keyword_sep = keyword_sep
        self.nlp = nlp
        
        # Define extensions
        Token.set_extension("is_stopword", default=None, force=True)
        Token.set_extension("is_punctuation", default=None, force=True)
        Token.set_extension("is_special_character", default=None, force=True)
        Token.set_extension("is_emoji", default=None, force=True)
        Token.set_extension("is_currency", default=None, force=True)
        Token.set_extension("is_digit", default=None, force=True)
        Token.set_extension("is_space", default=None, force=True)
        Token.set_extension("to_remove", default=None, force=True)
    
    def __call__(self, doc: Doc) -> Doc:
    
        string_list = self.doc_to_list(doc)
        filtered_string_list = self.filter(string_list)
        new_doc = self.list_to_doc(filtered_string_list)
    
        for token in new_doc:
        
            is_stopword = self.is_stopword(token.text.lower())
            is_punctutation = self.is_punctuation(token.text.lower())
            is_special_character = self.is_special_character(token.text.lower())
            is_emoji = self.is_emoji(token.text.lower())
            is_currency = self.is_currency(token.text.lower())
            is_digit = self.is_digit(token.text.lower())
            is_space = self.is_space(token.text.lower())
            
            to_remove = is_emoji or is_special_character or is_stopword or is_punctutation or is_currency or is_digit
        
            # Set extension values for each token
            token._.set("is_stopword", is_stopword)
            token._.set("is_punctuation", is_punctutation)
            token._.set("is_special_character", is_special_character)
            token._.set("is_emoji", is_emoji)
            token._.set("is_currency", is_currency)
            token._.set("is_digit", is_digit)
            token._.set("is_space", is_space)
            token._.set("to_remove", to_remove)
            
        return new_doc

    def list_to_doc(self, string_list):
        nlp = spacy.blank("en")
        words = " ".join(string_list).split()  # Combine the strings into a single space-separated string
        doc = Doc(nlp.vocab, words=words)  # Process the combined string with the spaCy pipeline
        return doc

    def doc_to_list(self, doc, to_lower=True):
        if to_lower:
            return [token.text.lower() for token in doc]
        else:
            return [token.text for token in doc]

    def is_special_character(self, token):
        return len(self.filter_special_characters(token)) == 0

    def filter_special_characters(self, token):

        # filter keyword
        new_token = re.sub(self.keyword_sep, ' ', token)
        
        # filter all special characters
        new_token = re.sub('[^a-zA-Z0-9\s]', ' ', new_token)
        
        # filter unknown characters
        new_token = re.sub('�', ' ', new_token)
        
        # remove if is composed by only special characters
        new_token = re.sub('^[_\W]+$', " ", new_token)
        
        # remove special characters (any) within string
        # except - : /
        #new_token = re.sub('[^a-zA-Z0-9\s:/-]', '', new_token)

        # remove all single characters
        #new_token = re.sub(r'\s+[a-zA-Z]\s+', ' ', new_token)
        
        # Remove single characters from the start
        #new_token = re.sub(r'\^[a-zA-Z]\s+', ' ', new_token) 
        
        # Removing prefixed 'b'
        #new_word = re.sub(r'^b\s+', '', new_word)

        return new_token.strip()
    
    def is_emoji(self, token):
        return len(self.filter_emojis(token)) == 0
    
    def filter_emojis(self, token):

        EMOJI_PATTERN = re.compile(
            "["
            "\U00002500-\U00002BEF"  # chinese char
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642" 
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+",
            flags=re.UNICODE
        )
        
        new_token = EMOJI_PATTERN.sub(' ', token.strip())
        
        return new_token.strip()
    
    def is_stopword(self, token):
        return len(self.filter_stopwords(token)) == 0
    
    def filter_stopwords(self, token):
        return "" if self.nlp.vocab[token].is_stop else token.strip()

    def is_punctuation(self, token):
        return len(self.filter_punctuations(token)) == 0

    def filter_punctuations(self, token):
        return "" if self.nlp.vocab[token].is_punct else token.strip()
        
    def is_digit(self, token):
        return len(self.filter_digits(token)) == 0
    
    def filter_digits(self, token):
        
        #'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    
        return "" if token.isdigit() else token.strip()

    def is_currency(self, token):
        return len(self.filter_currency(token)) == 0

    def filter_currency(self, token):
        return "" if self.nlp.vocab[token].is_currency else token.strip()
    
    def is_space(self, token):
        return len(self.filter_spaces(token)) == 0
    
    def filter_spaces(self, token):
        
        # removing spaces in front and back
        new_token = token.strip()
        
        # reduce spaces to just inside sentences
        new_token = re.sub(r'\s+', ' ', new_token)
        
        return new_token.strip()
        
    def break_subTokens(self, tokens):
        known_tokens = []
        
        for token in tokens:
            sub_tokens = token.split()
            known_tokens.extend(sub_tokens)
        
        return known_tokens
        
    def filter(self, tokens):
        """Filter stopwords, emojis, special characters, and punctuation from phrase.
        Args:
            tokens (list of str): List of tokens.
        """
        # Remove extra spaces
        filtered_tokens = copy.deepcopy([self.filter_spaces(token) for token in tokens if len(token)>0])
        filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )
            
        # Remove special characters
        if not self.keep_specials:
            filtered_tokens = copy.deepcopy([self.filter_special_characters(token) for token in filtered_tokens if len(token)>0])
            filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )
        
        # Remove emojis
        if not self.keep_emojis:
            filtered_tokens = copy.deepcopy([self.filter_emojis(token) for token in filtered_tokens if len(token)>0])
            filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )
        
        # Filter stopwords
        if not self.keep_stopwords:
            filtered_tokens = copy.deepcopy([self.filter_stopwords(token) for token in filtered_tokens if len(token)>0])
            filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )
            
        # Remove punctuations
        if not self.keep_puncts:
            filtered_tokens = copy.deepcopy([self.filter_punctuations(token) for token in filtered_tokens if len(token)>0])
            filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )
        
        # Remove digits
        if not self.keep_digits:
            filtered_tokens = copy.deepcopy([self.filter_digits(token) for token in filtered_tokens if len(token)>0])
            filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )
        
        # Remove currency
        if not self.keep_currency:
            filtered_tokens = copy.deepcopy([self.filter_currency(token) for token in filtered_tokens if len(token)>0])
            filtered_tokens = copy.deepcopy( self.break_subTokens(filtered_tokens) )

        # Remove '' characters
        filtered_tokens = [token for token in filtered_tokens if len(token)>0]

        return filtered_tokens

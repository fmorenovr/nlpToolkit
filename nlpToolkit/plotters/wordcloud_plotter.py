import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import copy

from ..text_processing import TextProcesser
from nltk import FreqDist
from wordcloud import WordCloud

class WordCloudPlotter:
    """WordCloud drawer."""
    def __init__(self, language_to_process=None, 
                        max_words=10000, 
                        mask=None, 
                        width=1000, 
                        height=1000, 
                        min_font_size=10, 
                        max_font_size=300, 
                        figsize=(20,10), 
                        to_gray_scale=False,
                        background_color="white", 
                        random_state=42):
        """WordCloud Drawer.
        Args:
            max_words (int): Number of words to show.
            mask (func): Area to draw.
            width (int): width of the output image.
            height (int): height of the output image.
            figsize (tuple): output image size.
            to_gray_scale (bool): draw in gray/rgb
        """
        self.language_to_process = language_to_process
        self.txt_processer = TextProcesser(language_to_process = language_to_process, log_tqdm=False)
        self.max_words = max_words
        self.mask = mask
        self.width = width
        self.height = height
        self.min_font_size=min_font_size
        self.max_font_size=max_font_size
        self.figsize = figsize
        self.background_color=background_color
        self.to_gray_scale = to_gray_scale
        self.random_state = random_state

        if self.mask is "circular":
            self.mask = self.generate_circular_mask()
        elif self.mask is "diamond":
            self.mask = self.generate_diamond_mask()

        self.wc_plotter = WordCloud(min_font_size=self.min_font_size,  
                                    max_font_size=self.max_font_size, 
                                    background_color=self.background_color, 
                                    max_words=self.max_words, 
                                    mask=self.mask, 
                                    contour_color='steelblue', 
                                    random_state=self.random_state,
                                    )

    # set nlp language
    def set_nlp_language(self, language_to_process):
        self.txt_processer.set_nlp_language(language_to_process=language_to_process)

    # return gray scale
    def to_grey_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

    # return circular area
    def generate_circular_mask(self):
        x, y = np.ogrid[:self.width, :self.height]
        mask = (x - self.width/2) ** 2 + (y - self.width/2) ** 2 > (self.width/2) ** 2
        mask = 255 * mask.astype(int)
        return mask
    
    # return diamond area    
    def generate_diamond_mask(self):
        x, y = np.ogrid[:self.width, :self.height]
        mask = abs(x-self.width/2) + abs(y-self.width/2) > self.width/2
        mask = 255 * mask.astype(int)
        return mask
        
    #def generate_square_mask(self):
    #    x, y = np.ogrid[:self.width, :self.height]
    #    mask = max(abs(x - self.width/2), abs(y - self.width/2)) > (self.width/2)
    #    mask = 255 * mask.astype(int)
    #    return mask

    # return words freq dict
    def process_texts_by_language(self, texts, language_to_process="english"):
        
        self.set_nlp_language(language_to_process=language_to_process)
        
        # Process and return word list
        _ = self.txt_processer.process_texts_by_language(texts)
        word_list = self.txt_processer.get_wordlist_from_processed_texts_by_language() 
        
        return word_list
    
    def calculate_words_frequency_by_language(self, texts, language_to_process="english"):

        word_list = self.process_texts_by_language(texts, language_to_process)

        self.freq_dict = FreqDist(word_list)
    
    def process_wordlist(self, tokens, languages_list=[]):
    
        if type(languages_list)==str:
            languages_list = [languages_list]
            
        word_list = self.txt_processer.process_wordlist(tokens, languages_list)
        
        return word_list
    
    def calculate_words_frequency(self, texts, languages_list=[]):

        texts_tokens = [txt.split(" ") for txt in texts]
        word_list = list(itertools.chain(*texts_tokens))

        if len(languages_list)>0:
            word_list = self.process_wordlist(word_list, languages_list)

        self.freq_dict = FreqDist(word_list)
        
    def get_freq_dict(self):
        return self.freq_dict
    
    def remove_words_from_dict(self, rm_list=[]):
        freq_dict = copy.deepcoty(self.get_freq_dict())
        for key in rm_list:
            freq_dict.pop(key, None)
        return freq_dict
            
    def plotWordCloud(self, title): 

        self.wc_plotter.generate_from_frequencies(self.freq_dict)

        fig = plt.figure(figsize=self.figsize)
        plt.axis("off")
        plt.title(title)
        if self.to_gray_scale:
            plt.imshow(self.wc_plotter.recolor(color_func=self.to_grey_color_func, random_state=3), interpolation="bilinear")
        else:
            plt.imshow(self.wc_plotter, interpolation="bilinear")
        plt.show()

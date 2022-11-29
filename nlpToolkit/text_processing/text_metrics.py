import enchant
import numpy as np
from functools import lru_cache
from sklearn.metrics import pairwise

class TextMetric:
    def __init__(self, language_to_eval=None, 
                       rounder=4):
        self.rounder=rounder
        
        self.lang_dict = {"portuguese":"pt_BR",
             "deutsche": "de_DE",
             "german": "de_DE",
             "english": "en_US",
             "russian": "ru_RU",
            }
        
        self.language_to_eval=language_to_eval
        if language_to_eval is not None:
            self.dict_lang = enchant.Dict(self.lang_dict[self.language_to_eval])

    def set_nlp_language(self, language_to_eval):
        self.language_to_eval = language_to_eval
        self.dict_lang = enchant.Dict(self.lang_dict[self.language_to_eval])

    # numerical values
    def dot_product(self, vecA, vecB):
        
        returned_value = np.dot(vecA, vecB)
        return round(returned_value, self.rounder)
        
    def cosine_similarity(self, vecA, vecB):
        similarity = pairwise.cosine_similarity([vecA], [vecB])
        
        returned_value = similarity[0][0]
        return round(returned_value, self.rounder)
    
    def cosine_distance(self, vecA, vecB):
        distance = pairwise.cosine_distances([vecA], [vecB])
        
        returned_value = distance[0][0]
        return round(returned_value, self.rounder)
    
    def angular_distance(self, vecA, vecB):
        similarity = pairwise.cosine_similarity([vecA], [vecB])
        similarity = similarity[0][0]
        arccos_value = np.arccos(similarity)
        if np.min(vecA)>=0.0 and np.min(vecB)>=0.0:
            arccos_value = 2.0*arccos_value

        returned_value = arccos_value/np.pi
        return round(returned_value, self.rounder)
        
    def angular_similarity(self, vecA, vecB):
    
        returned_value = 1.0 - self.angular_distance(vecA, vecB)
        return round(returned_value, self.rounder)

    # sets/text values
    def jaccard_similarity(txtA, txtB):
        #Find intersection of two sets
        nominator = set(txtA).intersection(set(txtB))

        #Find union of two sets
        denominator = set(txtA).union(set(txtB))
        
        #Take the ratio of sizes
        returned_value = len(nominator)/len(denominator)
        return round(returned_value, self.rounder)

    def jaccard_distance(self, txtA, txtB):
        #Find symmetric difference of two sets
        nominator = set(txtA).symmetric_difference(set(txtB))

        #Find union of two sets
        denominator = set(txtA).union(set(txtB))

        #Take the ratio of sizes
        returned_value = len(nominator)/len(denominator)
        return round(returned_value, self.rounder)

    # word values
    def levenshtein_distance(self, tokenA, tokenB):
        
        @lru_cache(None)  # for memorization
        def min_dist(s1, s2):

            if s1 == len(tokenA) or s2 == len(tokenB):
                return len(tokenA) - s1 + len(tokenB) - s2

            # no change required
            if tokenA[s1] == tokenB[s2]:
                return min_dist(s1 + 1, s2 + 1)

            return 1 + min(
                min_dist(s1, s2 + 1),      # insert character
                min_dist(s1 + 1, s2),      # delete character
                min_dist(s1 + 1, s2 + 1),  # replace character
            )

        return min_dist(0, 0)

    def init_levenshtein_matrix(self, tokenA, tokenB):
        distances = np.zeros((len(tokenA) + 1, len(tokenB) + 1))

        for t1 in range(len(tokenA) + 1):
            distances[t1][0] = t1

        for t2 in range(len(tokenB) + 1):
            distances[0][t2] = t2
        
        return distances
    
    def print_levenshtein_matrix(self, distances, tokenA, tokenB):
        tokenALength = len(tokenA)
        tokenBLength = len(tokenB)
        for t1 in range(tokenALength + 1):
            for t2 in range(tokenBLength + 1):
                print(int(distances[t1][t2]), end=" ")
            print()
        print("\n")

    def levenshtein_distance_matrix(self, tokenA, tokenB, log=False):
        distances = self.init_levenshtein_matrix(tokenA, tokenB)

        a = 0
        b = 0
        c = 0
        
        for t1 in range(1, len(tokenA) + 1):
            for t2 in range(1, len(tokenB) + 1):
                if (tokenA[t1-1] == tokenB[t2-1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]
                    
                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        if log:
            self.print_levenshtein_matrix(distances, tokenA, tokenB)
        
        return distances[len(tokenA)][len(tokenB)]

    def get_most_similar_word(self, word, log=False):
        
        suggest_list = self.dict_lang.suggest(word)
        
        levenshtein_dict = {}
        for suggest_word in suggest_list:
            distance = self.levenshtein_distance_matrix(suggest_word, word, log=log)
            levenshtein_dict[suggest_word] = distance
        
        closest_word = min(levenshtein_dict, key=levenshtein_dict.get)
        return closest_word, levenshtein_dict[closest_word], dict(sorted(levenshtein_dict.items(), key=lambda item: item[1]))

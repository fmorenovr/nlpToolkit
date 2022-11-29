import re
from lime.lime_text import LimeTextExplainer

# (?:(?:\s*\n\s*){2,}|(?<=[\.:])\s*\n\s*)
# (?:(?:\s*\n\s*){2,}) -> match 2 or more newlines
# (?<=[\.:])\s*\n\s* -> match a newline after a dot or a colon

# The idea is to divide the text into paragraphs, however there are cases in
# which the paragraphs are not divided by "\n\n", so we have to divide using
# "\.\n" without removing the dot (the same for colon).

class TextExplainer: # Binary classifiers
    def __init__(self, explainer_name="LIME",
                       num_features=10, 
                       num_samples=1000, 
                       random_state=42):
                       
        self.explainer_name = explainer_name
        self.num_features = num_features
        self.num_samples = num_samples
        self.split_regex = "(?:(?:\s*\n\s*){2,}|(?<=[\.:])\s*\n\s*)"
        
        if self.explainer_name.lower() == "lime":
            self.explainer = LimeTextExplainer(
                                class_names=["Not theme", "Theme"],
                                split_expression=lambda x: re.split(self.split_regex, x),
                                bow=False,
                                mask_string="",
                                random_state=random_state,
                            )
        else:
            self.explainer = LimeTextExplainer(
                                class_names=["Not theme", "Theme"],
                                split_expression=lambda x: re.split(self.split_regex, x),
                                bow=False,
                                mask_string="",
                                random_state=random_state,
                            )

    def get_explanation(self, text, predict_proba):
        if self.explainer_name.lower() == "lime":
            return self.lime_explanation(text, predict_proba)
        else:
            return self.lime_explanation(text, predict_proba)

    def lime_explanation(self, text, predict_proba):
        """Generate explanation for a text using LIME.
        Args:
            text (str): Text to be explained.
            predict_proba (function): Function that receives a text and returns a
                probability of the theme being applied to the text.
        """
        explanation = self.explainer.explain_instance(
                              text,
                              predict_proba,
                              (1,), # labels
                              num_features=self.num_features,
                              num_samples=self.num_samples,
                          )

        paragraphs = re.split(self.split_regex, text)
        ids_to_scores = self.lime_explanation_to_scores(explanation, 1)
        scores = [0]*len(paragraphs)
        for id_, score in ids_to_scores.items():
            scores[id_] = round(score, 4)

        return scores

    def lime_explanation_to_scores(self, explanation, label):
        ids_n_scores = explanation.local_exp[label]
        return dict(ids_n_scores)

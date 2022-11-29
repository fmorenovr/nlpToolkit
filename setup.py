from setuptools import setup
from pathlib import Path

with open(Path('./requirements.txt').resolve()) as f:
    required = f.read().splitlines()

setup(
    name="nlpToolkit",  
    packages=["nlpToolkit", "nlpToolkit/language_processing", "nlpToolkit/text_processing", "nlpToolkit/plotters"],
    install_requires=required,
    version="0.1.1"
)



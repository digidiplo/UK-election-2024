import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import nltk
import spacy

print("All libraries installed successfully!")

# Download NLTK data if not done already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
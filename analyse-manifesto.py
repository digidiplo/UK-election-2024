import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yaml
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
import spacy
from typing import List, Dict
import string

from wordcloud import WordCloud
import os

# Download NLTK data if necessary
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 8000000  # Increase the max length if necessary

def get_words(text: str) -> List[str]:
    """
    Get every word in the text that isn't a stopword or punctuation,
    and that is either a noun, adjective, verb or interjection
    (based on the universal POS tags)
    Args:
        text (str): Text to be processed.
    Returns:
        List[str]: List of words.
    """
    doc = nlp(text)
    words = [
        word.text.replace("\n", "").replace("*", "")  # remove new line and bold symbols
        for word in doc
        if not word.is_stop  # remove stopwords
        and not word.is_punct  # remove punctuation
        and (
            word.pos_ == "NOUN"  # noun
            or word.pos_ == "ADJ"  # adjective
            or word.pos_ == "VERB"  # verb
            or word.pos_ == "INTJ"  # interjection
        )
    ]
    # remove blank words and spaces
    words = [word for word in words if word != ""]
    words = [word.replace(" ", "") for word in words]
    # make all words lowercase
    words = [word.lower() for word in words]
    # remove words with less than 3 characters
    words = [word for word in words if len(word) > 2]
    return words


def preprocess_corpus(corpus: List[str]) -> List[str]:
    preprocessed_corpus = []
    for doc in corpus:
        words = get_words(doc)
        preprocessed_doc = ' '.join(words)
        preprocessed_corpus.append(preprocessed_doc)
    return preprocessed_corpus


def _add_sentence_to_list(sentence: str, sentences_list: List[str]):
    """
    Add a sentence to the list of sentences.
    Args:
        sentence (str): Sentence to be added.
        sentences_list (List[str]): List of sentences.
    """
    while sentence.startswith(" "):
        sentence = sentence[1:]  # remove leading space
    if all(c in string.punctuation for c in sentence) or len(sentence) == 1:
        return  # skip sentences with only punctuation
    sentences_list.append(sentence)


def get_sentences(text: str) -> List[str]:
    """
    Get sentences from a text.
    Args:
        text (str): Text to be processed.
    Returns:
        List[str]: List of sentences.
    """
    paragraphs = text.split("\n")
    paragraphs = [p for p in paragraphs if p != ""]
    sentences = list()
    
    for paragraph in paragraphs:
        if paragraph.startswith("#"):  # treat headings as sentences
            _add_sentence_to_list(paragraph, sentences)
            continue
        
        prev_sentence_idx = 0
        for idx in range(len(paragraph)):
            if idx + 1 < len(paragraph):
                if (paragraph[idx] == "." and not paragraph[idx + 1].isdigit()) or (paragraph[idx] in "!?" or paragraph[idx] in string.punctuation and paragraph[idx+1] == " "):
                    sentence = paragraph[prev_sentence_idx : idx + 1]
                    _add_sentence_to_list(sentence, sentences)
                    prev_sentence_idx = idx + 1
            else:
                sentence = paragraph[prev_sentence_idx:]
                _add_sentence_to_list(sentence, sentences)
    
    return sentences

def get_word_cloud(
    words: List[str],
    max_words: int = 500,
    image_path: str = None,
    image_name: str = None,
):
    """
    Create a word cloud based on a set of words.
    Args:
        words (List[str]):
            List of words to be included in the word cloud.
        max_words (int):
            Maximum number of words to be included in the word cloud.
        image_path (str):
            Path to the image file where to save the word cloud.
        image_name (str):
            Name of the image where to save the word cloud.
    """
    # change the value to black
    def black_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        return "hsl(0,100%, 1%)"

    # set the wordcloud background color to white
    # set width and height to higher quality, 3000 x 2000
    wordcloud = WordCloud(
        font_path="/Library/Fonts/Arial Unicode.ttf",
        background_color="white",
        width=3000,
        height=2000,
        max_words=max_words,
    ).generate(" ".join(words))
    # set the word color to black
    wordcloud.recolor(color_func=black_color_func)
    # set the figsize
    plt.figure(figsize=[15, 10])
    # plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # remove plot axes
    plt.axis("off")
    if image_path is not None and image_name is not None:
        # save the image
        plt.savefig(os.path.join(image_path, image_name), bbox_inches="tight")


def get_topical_sentences(
    sentences: List[str], topics: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Get lists of sentences per topic, based on the presence of
    words that are a part of the topic.
    Args:
        sentences (List[str]):
            List of sentences to analyse.
        topics (Dict[str, List[str]]):
            Dictionary of words per topic.
    Returns:
        Dict[str, List[str]]:
            Dictionary of sentences per topic.
    """
    topical_sentences = dict()
    for topic in topics:
        topical_sentences[topic] = list()
    for sentence in sentences:
        for topic in topics:
            if any(topical_word in sentence.lower() for topical_word in topics[topic]):
                topical_sentences[topic].append(sentence)
    return topical_sentences


# Directory containing the manifesto .md files
manifesto_dir = "manifestos"

# Get a list of all .md files in the directory
manifesto_files = [file for file in os.listdir(manifesto_dir) if file.endswith(".md")]

# Read the content of each manifesto file
corpus = []
for file in manifesto_files:
    with open(os.path.join(manifesto_dir, file), 'r') as f:
        content = f.read()
        corpus.append(content)

# Preprocess the corpus
preprocessed_corpus = preprocess_corpus(corpus)

# Create the TF-IDF Vectorizer with the specified settings
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

# Fit and transform the preprocessed corpus
X = vectorizer.fit_transform(preprocessed_corpus)

# Extract the feature names (words) from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense matrix
dense = X.todense()

# Convert the dense matrix to a list
denselist = dense.tolist()

# Create a DataFrame from the dense list with feature names as columns
df = pd.DataFrame(denselist, columns=feature_names)

# Transpose the DataFrame
data = df.transpose()

# Assign column names to the transposed DataFrame
data.columns = [file[:-3] for file in manifesto_files]  # Remove the '.md' extension from file names

# Find the top 30 words said by each President
top_dict = {}
for c in range(len(data.columns)):
    top = data.iloc[:, c].sort_values(ascending=False).head(30)
    top_dict[data.columns[c]] = list(zip(top.index, top.values))

# Print the top 15 words said by each Party
# for document, top_words in top_dict.items():
  #  print(document)
  #  print(', '.join([word for word, count in top_words[0:15]]))
  #  print('---')

# Generate word cloud for a specific document
# document_name = 'Greens'  # Replace with the desired document name

# Define the black color function
# def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
  #  return "hsl(0,100%, 1%)"

# Generate the word cloud with customized settings
# wordcloud = WordCloud(font_path='/Library/Fonts/Arial Unicode.ttf',
                    #  background_color="white",
                    #  width=3000, height=2000,
                    #  max_words=500).generate_from_frequencies(data[document_name])

# Set the word color to black
# wordcloud.recolor(color_func=black_color_func)

# Plot the word cloud
# plt.figure(figsize=(15, 10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title(f'Word Cloud for {document_name}')
# plt.savefig(f'{document_name}_wc.png') # Save the image
# plt.show()

with open('topic_keywords.yaml', 'r') as file:
    topic_keywords = yaml.safe_load(file)

with open('manifestos/Labour.md', 'r') as file:
    text = file.read()

# Get the sentences from the text
sentences = get_sentences(text)

# Get the topical sentences
topical_sentences = get_topical_sentences(sentences, topic_keywords)

# Print the topical sentences for each topic
for topic, sentences in topical_sentences.items():
    print(f"Topic: {topic}")
    for sentence in sentences:
        print(f"- {sentence}")
    print()
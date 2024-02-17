## Semantic Signal Separation - From https://medium.com/towards-data-science/semantic-signal-separation-769f43b46779
## Semantic Signal Separation (SSS) is a statistical model inspired by classical topic models like Latent Semantic Allocation (LSA) 
## but incorporates principles from Independent Component Analysis (ICA) to extract maximally independent semantic components from text.
## And SSS model is a statistical model that seeks to uncover maximally independent semantic components in a corpus of text data. 
## It uses principles from Independent Component Analysis (ICA) to decompose the representations of these components and identify the words 
## that are most strongly associated with each component

## Example
!pip install turftopic datasets
from datasets import load_dataset

ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")

from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(10, encoder="all-MiniLM-L12-v2")
model.fit(ds["abstract"])

model.print_topics()

import numpy as np

vocab = model.get_vocab()

# We will produce a BoW matrix to extract term frequencies
document_term_matrix = model.vectorizer.transform(ds["abstract"])
frequencies = document_term_matrix.sum(axis=0)
frequencies = np.squeeze(np.asarray(frequencies))

# We select the 99th percentile
selected_terms_mask = frequencies > np.quantile(frequencies, 0.99)

import pandas as pd

# model.components_ is a n_topics x n_terms matrix
# It contains the strength of all components for each word.
# Here we are selecting components for the words we selected earlier

terms_with_axes = pd.DataFrame({
    "inference": model.components_[7][selected_terms],
    "measurement_devices": model.components_[1][selected_terms],
    "noise": model.components_[6][selected_terms],
    "term": vocab[selected_terms]
 })

import plotly.express as px

px.scatter(
    terms_with_axes,
    text="term",
    x="inference",
    y="noise",
    color="measurement_devices",
    template="plotly_white",
    color_continuous_scale="Bluered",
).update_layout(
    width=1200,
    height=800
).update_traces(
    textposition="top center",
    marker=dict(size=12, line=dict(width=2, color="white"))
)

### From https://medium.com/data-science-collective/k-nearest-rank-scores-a-tool-for-local-structure-analysis-in-embeddings-ee4cbbbb996d

"""
1. What the Article Is About
   -a. Big picture: When you squeeze very high-dimensional data (hundreds or thousands of columns) down to two or three columns for a scatter-plot,
                    you hope the picture still tells the truth about which samples are truly close to each other.
   -b. Reality: Every dimensionality-reduction tool—PCA, t-SNE, UMAP, etc.—distorts something.
   -c. Goal of the article:
       -1. Explain a metric called the K-Nearest-Rank Similarity score (KNR-score) that measures how well two maps (e.g. “50-D PCA” vs “2-D t-SNE”) agree about neighbor relationships.
       -2. Show, with runnable Python code, how to compute KNR-scores on the MNIST digit set and interpret the heat-map results.

2. Quick Primer on Dimensionality-Reduction Tools
   Type	| Examples	| How they basically work | 	Typical use
   Linear	| PCA (Principal Component Analysis)	| Builds new axes that are linear combinations of the original features and sorts them by “variance explained.”	| Feature ranking, speed-up, de-noising, sometimes 2-D visuals.
   Non-linear |	t-SNE, UMAP	Try to keep local neighbor relations intact, even if global distances warp.	| 2-D/3-D plots that reveal clusters.

   Key point: t-SNE/UMAP usually draw prettier, more separated clusters than PCA, but we need a quantitative way to see if that prettiness
              still respects the real neighbor structure.

3 K-Nearest-Rank Similarity (KNR-score) — Step by Step
  3.1 The Intuition
      -a. Pick a sample.
      -b. Ask each map: “Who are your K closest friends?”
      -c. If both maps name (mostly) the same friends, they agree at that scale.
      -d. Square that idea across all samples and scales → KNR-score matrix.
  3.2 The Four Mechanical Steps
      -a. Compute Euclidean distances inside each map.
      -b. Rank neighbors for every sample (rank 1 = closest).
      -c. For chosen K-values 𝑘_𝑥 and 𝑘_𝑦 in maps 𝑋 and 𝑌, compare the two rank lists.
      -d. Score the overlap; 1 = perfect agreement, 0 = no common neighbors.
  3.3 The Formula
      For sample 𝑖 and its neighbor list of length 𝑘_𝑥 in map 𝑋 and 𝑘_𝑦 in map 𝑌:
      𝑆_(𝑋,𝑌) (𝑘_𝑥,𝑘_𝑦)=1/𝑛 ∑(𝑖=1 to 𝑖=𝑛) 1[neighbor sets match]
      with indicator
      1[condition]={1  if condition true
                    0  otherwise

      A reddish cell in the eventual heat-map means “high similarity at those K-scales,” a bluish cell means “maps disagree.”

4. Hands-On Example with MNIST Digits
   All code below can be copied into a notebook exactly as shown.
"""
!pip install KNRscore d3blocks umap-learn

import numpy as np
from sklearn import manifold, decomposition
from umap import UMAP
import KNRscore as knrs

# 1 Load example 64-pixel digit images (8×8) and labels
X, y = knrs.import_example(data='digits')

# 2 Create four projections
map_pca  = decomposition.TruncatedSVD(n_components=50).fit_transform(X)       # 50-D
map_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(X)         # 2-D
map_umap = UMAP(densmap=True).fit_transform(X)                                # 2-D
map_rand = np.c_[np.random.permutation(map_tsne[:,0]),                        # shuffled control
                 np.random.permutation(map_tsne[:,1])]

# 3 Quick scatter-plots (static)
fig, ax = knrs.scatter(map_pca[:,0], map_pca[:,1], labels=y, title='PCA')
fig, ax = knrs.scatter(map_tsne[:,0], map_tsne[:,1], labels=y, title='t-SNE')
fig, ax = knrs.scatter(map_umap[:,0], map_umap[:,1], labels=y, title='UMAP')
fig, ax = knrs.scatter(map_rand[:,0], map_rand[:,1], labels=y, title='Random')

#### 4.1 Optional Interactive Scatter
from d3blocks import D3Blocks
d3 = D3Blocks()
d3.scatter(map_pca[:,0], map_pca[:,1],            # plot 1
           x1=map_tsne[:,0], y1=map_tsne[:,1],    # plot 2
           x2=map_umap[:,0], y2=map_umap[:,1],    # plot 3
           label_radio=['PCA','t-SNE','UMAP'],
           scale=True,
           tooltip=['ID '+str(i) for i in range(X.shape[0])],
           color=y.astype(int).astype(str),
           filepath='scatter_embeddings.html')

##### 5 Quantitative Comparisons with KNR-score
#### 5.1 50-D PCA vs 2-D t-SNE
scores = knrs.compare(map_pca, map_tsne, n_steps=5)
knrs.plot(scores, xlabel='PCA-50D', ylabel='t-SNE-2D')

## Interpretation: Heat-map is reddish almost everywhere → t-SNE kept both local and global neighbor relations remarkably well relative 
## to the full 50-D PCA space.

#### 5.2 2-D PCA vs 2-D t-SNE
scores = knrs.compare(map_pca[:,0:2], map_tsne, n_steps=5)
knrs.plot(scores, xlabel='PCA-2D', ylabel='t-SNE-2D')

## Interpretation: Cooler (green/blue) values, especially in the bottom-left (most local) corner → the first two PCA axes alone do not
## preserve the same close-neighbor structure as t-SNE.

#### 5.3 2-D UMAP vs 2-D t-SNE
scores = knrs.compare(map_umap, map_tsne, n_steps=5)
knrs.plot(scores, xlabel='UMAP-2D', ylabel='t-SNE-2D')

## Interpretation: Mostly warm colors → both nonlinear methods build similar overall clusters. 
## Slightly lower scores (≈0.5–0.6) for K = 6 signals they sometimes shuffle the exact ordering of the nearest six neighbors inside a cluster.

#### 5.4 2-D Random Shuffle vs 2-D t-SNE
scores = knrs.compare(map_rand, map_tsne, n_steps=5)
knrs.plot(scores, xlabel='Random-2D', ylabel='t-SNE-2D')

##Interpretation: Deep blue everywhere → as expected, random coordinates share virtually no neighbor structure with t-SNE.

"""
6. Key Take-Home Lessons
   -1. Always test whether a pretty 2-D plot is faithful to your real data geometry.
   -2. KNR-score gives a single glance (heat-map) answer for every “how many neighbors” scale.
   -3. t-SNE & UMAP preserved MNIST’s local structure well; 2-D PCA alone did not.
   -4. A projection is just a projection—check its stability before basing downstream decisions on cluster membership or distances.


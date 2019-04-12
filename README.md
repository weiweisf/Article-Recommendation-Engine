# Article-Recommendation-Engine

## Goal     
Make a simple article recommendation engine.         


## Method      
- To do this, use Stanfor's Glove to represent the word vectors (i.e. word2vec) and use BOW (Bag of Words) to represent every article with a single centroid vector in 300 dimensions.      
- Given an article, we can compute the distance from its centroid to every other article's centroid. The article centroids closest to the article of interest's centroid are the most similar articles.     



# Graph Embedding Methods

In the experiemtns of DSN, we use four graph embedding methods to generate node embedding of users in the social graph, namely
- topology-based graph embedding
    - DeepWalk
    - LINE
    - node2vec
- Influence-aware graph neural network for graph embedding
    - IMINFECTOR
    
The implementations of DeepWalk, LINE and node2vec are based on the code from [Graph Embedding](https://github.com/shenweichen/GraphEmbedding), and IMINFECTOR is based on the code from [IMINFECTOR](https://github.com/geopanag/IMINFECTOR).
 
Since the graph embedding part is not the core of DSN, but the preliminary work to project the users in social graph to vector space, here we provide node2vec process as an example of graph embedding part.
  


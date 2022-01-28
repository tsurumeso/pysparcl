# pysparcl

Python implementation of the sparse clustering methods of Witten and Tibshirani (2010).

## Demo results

Each sample has 1000 features, and 1 % of them are informative.

|Hierarchical clustering|Sparse hierarchical clustering|
|:-:|:-:|
|![](images/hc.png)|![](images/shc.png)|

## Functions

- Sparse hierarchical clustering
- Sparse KMeans clustering
- Selection of turning parameter for sparse hierarchical clustering
- Selection of turning parameter for sparse KMeans clustering

## Installation

### Getting pysparcl
```
git clone https://github.com/tsurumeso/pysparcl.git
```

### Run setup script
```
cd pysparcl
python setup.py install
```

### Run demo
Perform sparse hierarchical clustering.
```
cd demo
python run.py
```

Perform sparse KMeans clustering.
```
cd demo
python run.py -m kmeans
```

## Usage
```python
import matplotlib.pyplot as plt
import pysparcl

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


# X is a numpy array of (samples, features) shape.
perm = pysparcl.hierarchy.permute(X)
result = pysparcl.hierarchy.pdist(X, wbound=perm['bestw'])
link = linkage(result['u'], method='average')
dendro = dendrogram(link)
plt.show()
```

## References
- [1] D. M. Witten and R. Tibshirani, "A framework for feature selection in clustering",  
Am. Stat., vol. 105, no. 490, pp. 713–726, 2010.
- [2] "sparcl: Perform sparse hierarchical clustering and sparse k-means clustering",  
https://cran.r-project.org/web/packages/sparcl/index.html

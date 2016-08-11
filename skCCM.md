---
layout: page
title: skCCM
permalink: /skCCM/
---

**Scikit Convergent Cross Mapping**

Convergent cross mapping (ccm) is used as a way to detect causality in time series using ideas from [nonlinear analysis][wiki-nla]. The full paper, [Detecting Causality in Complex Ecosystems][paper] by Sugihara et al. is an interesting read and I recommend reading it. Additionally, there are a couple of good youtube videos by his lab group that explain ccm:

1. [Time Series and Dynamic Manifolds][vid-1]
2. [Reconstructed Shadow Manifold][vid-2]
3. [Convergent Cross Mapping][vid-3]

If you are interested, there is a [full talk][sug-talk] by Dr. Sugihara that extends some of these ideas into other domains.

**Package**

[skCCM][skccm] attempts to mimic the ease and style of scikit-learn's api. The package does stray a little from the `.fit` and `.predict` methods, but the aim is still on simplicity and ease of use.

**Installation**

`pip install skCCM`

***

# Quick Example

To illustrate how this package works, we will start with an example where the figures from the paper above will be reproduced. The coupled logistic map from the paper takes the form:

$$X(t+1) = X(t)[r_x - r_x X(t) - \beta_{x,y}Y(t)]$$

$$Y(t+1) = Y(t)[r_y - r_y Y(t) - \beta_{y,x}X(t)]$$

Notice that $$\beta_{x,y}$$ controls the amount of information from the $$Y$$ time series that is being injected into the $$X$$ time series. Likewise, $$\beta_{y,x}$$ controls the amount of information injected into the $$Y$$ time series from the $$X$$ time series. These parameters control how much one series influences the other. There is a function in `skCCM.data` to reproduce these time series. For example:

{% highlight python linenos %}
import skCCM.data as data

rx1 = 3.72 #determines chaotic behavior of the x1 series
rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = 0.2 #Influence of x1 on x2
b21 = 0.01 #Influence of x2 on x1
ts_length = 1000
x1,x2 = data.coupled_logistic(rx1,rx2,b12,b21,ts_length)
{% endhighlight %}

Using these parameters, `x1` has more of an influence on `x2` than `x2` has on `x1`. This produces the coupled logistic map as seen in the figure below where the top plot is `x1` and the bottom is `x2`.

![coupled_logistic](/assets/ccm/coupled_logistic.png){: .center-image }

As is clearly evident from the figure above, there is no way to tell if one series is influencing the other just by examining the time series.

The next step is to calculate the mutual information of the time series so that we can appropriately determine the lag value for the embedding. The first minimum in the mutual information can be thought of as jumping far enough away that there is new information there. This can be done using the `embed` class provided by skCCM.

{% highlight python linenos %}
import skCCM as ccm

e1 = ccm.embed(x1) #initiate the class
e2 = ccm.embed(x2)

mi1 = e1.mutual_information(10)
mi2 = e2.mutual_information(10)
{% endhighlight %}

The top plot below is `mi1` and the bottom is `mi2`
![mutual_info](/assets/ccm/mutual_info.png){: .center-image }

As is seen above, the mutual information is continually decreasing, so a lag of one is sufficient to rebuild a shadow manifold (or a Poincaré section since it is noncontinuous). For series that are not changing as rapidly from one timestep to the next, larger lag values will be indicated by the first minimum.

{% highlight python linenos %}
lag = 1
embed = 2
X1 = e1.embed_vectors_1d(lag,embed)
X2 = e2.embed_vectors_1d(lag,embed)
{% endhighlight %}

![x_embedded](/assets/ccm/x_embedded.png){: .center-image }

Now that we have embedded the time series, all that is left to do is check the forecast skill as a function of library length. This package diverges from the paper above in that a training set is used to rebuild the shadow manifold and the testing set is used to see if nearby points on one manifold can be used to make accurate predictions about the other manifold. This removes the problem of autocorrelated time series.

{% highlight python linenos %}
from skCCM.utilities import train_test_split

#split the embedded time series
x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM = ccm.ccm() #initiate the class

#library lengths to test
len_tr = len(x1tr)
lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')

#test causation
sc1, sc2 = CCM.predict_causation(x1tr, x1te, x2tr, x2te,lib_lens)
{% endhighlight %}

![xmap_lib_len](/assets/ccm/xmap_lib_len.png){: .center-image }

As can be seen from the image above, `x1` has a higher prediction skill. Another way to view this is that information about `x1` is present in the `x2` time series. This leads to better forecasts for `x1` using `x2`'s reconstructed manifold. This means that `x1` is driving `x2` which is exactly how we set the initial conditions when we generated these time series.

To make sure that this algorithm is robust we test a range of $$\beta$$ values similar to how the original paper. The results below show the difference between `sc1` and `sc2`.

![xmap_changingB](/assets/ccm/xmap_changingB.png){: .center-image }



# Thorough Explanation

#### class **embed**

{% highlight python linenos %}
def __init__(self, X):
  """
  Parameters
  ----------
  X : series or dataframe,
  """

def mutual_information(self, max_lag):
  """
  Calculates the mutual information between an unshifted time series
  and a shifted time series. Utilizes scikit-learn's implementation of
  the mutual information found in sklearn.metrics.

  Parameters
  ----------
  max_lag : integer
    maximum amount to shift the time series

  Returns
  -------
  m_score : 1-D array
    mutual information between the unshifted time series and the
    shifted time series
  """

def embed_vectors_1d(self, lag, embed):
  """
  Embeds vectors from a one dimensional time series in
  m-dimensional space.

  Parameters
  ----------
  X : array
    A 1-D array representing the training or testing set.

  lag : int
    lag values as calculated from the first minimum of the mutual info.

  embed : int
    embedding dimension, how many lag values to take

  predict : int
    distance to forecast (see example)


  Returns
  -------
  features : array of shape [num_vectors,embed]
    A 2-D array containing all of the embedded vectors

  Example
  -------
  X = [0,1,2,3,4,5,6,7,8,9,10]

  em = 3
  lag = 2

  returns:
  features = [[0,2,4], [1,3,5], [2,4,6], [3,5,7]]
  """
{% endhighlight %}


#### class **ccm**

{% highlight python linenos %}

def __init__(self, weights='exponential_paper', verbose=False,
		score_metric='corrcoef' ):
  """
  Parameters
  ----------
  weights : weighting scheme for predictions
    - exponential_paper : weighting scheme from paper
  verbose : prints out calculation status
  score : how to score the predictions
    -'score'
    -'corrcoef'
  """

def predict_causation(self,X1_train,X1_test,X2_train,X2_test,lib_lens):
  """
  Wrapper for predicting causation as a function of library length.
  X1_train : embedded train series of shape (num_samps,embed_dim)
  X2_train : embedded train series of shape (num_samps,embed_dim)
  X1_test : embedded test series of shape (num_samps, embed_dim)
  X2_test : embedded test series of shape (num_samps, embed_dim)
  lib_lens : which library lengths to use for prediction
  near_neighs : how many near neighbors to use (int)
  how : how to score the predictions
    -'score'
    -'corrcoef'
  """

def fit(self,X1_train,X2_train):
  """
  Fit the training data for ccm. Amount of near neighbors is set to be
  embedding dimension plus one. Creates seperate near neighbor regressors
  for X1 and X2 independently. Also Calculates the distances to each
  sample.

  X1 : embedded time series of shape (num_samps,embed_dim)
  X2 : embedded time series of shape (num_samps,embed_dim)
  near_neighs : number of near neighbors to use
  """

def dist_calc(self,X1_test,X2_test):
  """
  Calculates the distance from X1_test to X1_train and X2_test to
  X2_train.

  Returns
  -------
  dist1 : distance from X1_train to X1_test
  ind1 : indices that correspond to the closest
  dist2 : distance from X2_train to X2_test
  ind2 : indices that correspond to the closest
  """

def weight_calc(self,d1,d2):
  """
  Calculates the weights based on the distances.
  Parameters
  ----------
  d1 : distances from X1_train to X1_test
  d2 : distances from X2_train to X2_test
  """

def predict(self,X1_test,X2_test):
  """
  Make a prediction

  Parameters
  ----------
  X1 : test set
  X2 : test set
  """

def score(self):
  """
  Evalulate the predictions
  how : how to score the predictions
    -'score'
    -'corrcoef'
  """
{% endhighlight %}








[wiki-nla]: https://www.wikiwand.com/en/Nonlinear_functional_analysis
[vid-1]: https://www.youtube.com/watch?v=fevurdpiRYg
[vid-2]: https://www.youtube.com/watch?v=rs3gYeZeJcw
[vid-3]: https://www.youtube.com/watch?v=iSttQwb-_5Y
[sug-talk]: https://www.youtube.com/watch?v=uhONGgfx8Do
[paper]: http://science.sciencemag.org/content/338/6106/496
[skccm]:https://github.com/NickC1/skCCM
[pandas-datareader]: https://github.com/pydata/pandas-datareader

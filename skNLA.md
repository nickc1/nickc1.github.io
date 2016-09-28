---
layout: page
title: skNLA
permalink: /skNLA/
---

**Scikit Nonlinear Analysis**

Scikit Nonlinear Analysis (nla) can be used as a way to forecast time series, spatio-temporal images, and even discrete spatial arangements. More importantly, skNLA can provide insite into the underlying dynamics of a system. For a more complete background, I suggest checking out the [Nonlinear Analysis by Kantz][nla-book]. For a more brief overview the wikipedia article on [nonlinear analysis][wiki-nla] is a good start. Additionally, Dr. Sugihara's lab has produced some good summary videos of the topic:

1. [Time Series and Dynamic Manifolds][vid-1]
2. [Reconstructed Shadow Manifold][vid-2]

**Package**

[skNLA][sknla] attempts to mimic the ease and style of scikit-learn's api. The package does stray a little from the `.fit` and `.predict` methods, but the aim is still on simplicity and ease of use.

**Installation**

`pip install skNLA`

***
<br>

# Quick Example

In order to illustrate how this package works, we start with an example as outlined in the paper above. The [lorenz system][lorenz-wiki] takes the form of :

$$\frac{dx}{dt} = \sigma (y - x)$$

$$ \frac{dy}{dt} = x(\rho - z) - y$$

$$\frac{dz}{dt} = xy - \beta z$$

Here, we are going to make forecasts of the $$x$$ time series. Note that this series, while completely deterministic, is a classic [chaotic system][chaos-wiki]. This means that making forecasts into the future is going to be difficult as small changes in the initial conditions can lead to drastically different trajectories of the system.

There is a function in `skNLA.data` to reproduce these time series. For example:

{% highlight python linenos %}
import skNLA.data as data

X = data.lorenz()[:,0] #only going to use the x values
{% endhighlight %}

![coupled_logistic](/assets/nla/lorenz.png){: .center-image }

The next step is to calculate the mutual information of the time series so that we can appropriately determine the lag value for the embedding. The first minimum in the [mutual information][mutual-info-wiki] can be thought of as jumping far enough away that there is new information gained. A more useful thought construct might be to think of it as the first minimum in the autocorrelation. Mutual information, however, has proved to be more useful in appropriately picking the lag. [cite] The mutual information calculation can be done using the `embed` class provided by skNLA.

{% highlight python linenos %}
import skNLF as nlf

E = nlf.embed(X) #initiate the class

max_lag = 100
mi = E.mutual_information(max_lag)
{% endhighlight %}

![mutual_info](/assets/nla/lorenz_mutual_info.png){: .center-image }

As seen above, the first minimum of the mutual information is at lag=18. This is the lag that will be used to rebuild a shadow manifold. This is done by:

{% highlight python linenos %}
lag = 18
embed = 3
predict = 36 #predicting out to double to lag
X,y = E.embed_vectors_1d(lag,embed,predict)
{% endhighlight %}

![x_embedded](/assets/nla/embedded_lorenz.png){: .center-image }

Now that we have embedded the time series, all that is left to do is check the forecast skill as a function of library length. First we split it into a training set and testing set. Additionally, we will initiate the class.

{% highlight python linenos %}
#split it into training and testing sets
train_len = int(.75*len(X))
Xtrain = X[0:train_len]
ytrain = y[0:train_len]
Xtest = X[train_len:]
ytest = y[train_len:]

max_nn = .1 * len(Xtrain) # test out to a maximum of 10% of possible NN
weights = 'distance' #use a distance weighting for the near neighbors
NLF = nlf.NonLin(max_nn,weights) # initiate the nonlinear forecasting class

{% endhighlight %}

The next step is to then fit the data and calculate the distance from the training set to the testing set.

{% highlight python linenos %}
NLF.dist_calc(Xtest) #calculate the distance all the near neighbors

nn_range = np.arange(1,max_nn,10,dtype='int')
preds = NLF.predict_range(nn_range)

s_range = NLF.score_range(ytest) #score
{% endhighlight %}

![xmap_lib_len](/assets/nla/lorenz_forecast_range.png){: .center-image }

As can be seen from the image above, the highest forecast skill is located at low numbers of near neighbors and low forecast distances.

***
<br>

# Two-Dimensional Extension

This algorithm can be extended to two dimensions. The only thing that changes is the embedding algorithm. Everything else works exactly the same. We will explore a spatio-temporal logistic map that is diffused in space.

{% highlight python linenos %}
X = data.chaos2D(sz=256)
{% endhighlight %}


![xmap_lib_len](/assets/nla/2d_chaos.png){: .center-image }

Next, the mutual information along the rows and down the columns is calculated.

{% highlight python linenos %}
E = nlf.embed(X)
rmi,cmi,rmut,cmut = E2.mutual_information_spatial(30)
{% endhighlight %}

![xmap_lib_len](/assets/nla/2d_chaos_mutual_info.png){: .center-image }

This hints that the system has a mutual information of 6 along the rows and 5 down the columns.

{% highlight python linenos %}
lag = (6,5)
embed= (2,3)
predict = 10

X,y = E.embed_vectors_2d(lag,embed,predict)
{% endhighlight %}

After the series has been embedded, the procedure is the same as above. Next we split it into a training set and testing set and initiate the class.

{% highlight python linenos %}
#split it into training and testing sets
train_len = int(.75*len(X))
Xtrain = X[0:train_len]
ytrain = y[0:train_len]
Xtest = X[train_len:]
ytest = y[train_len:]

max_nn = .1 * len(Xtrain) # test out to a maximum of 10% of possible NN
weights = 'distance' #use a distance weighting for the near neighbors
NLF = nlf.NonLin(max_nn,weights) # initiate the nonlinear forecasting class
{% endhighlight %}

Next we fit the model, calculate the distances from the training set to the testing set, and finally make some predictions.

{% highlight python linenos %}
NLF.fit(Xtrain,ytrain) #fit the training data

NLF.dist_calc(Xtest) #calculate the distance all the near neighbors

nn_range = np.arange(1,max_nn,10,dtype='int')
preds = NLF.predict_range(nn_range)
{% endhighlight %}

Next, we score the predictions and visualize in a contour plot.

{% highlight python linenos %}
s_range = NLF.score_range(ytest)
{% endhighlight %}

![xmap_lib_len](/assets/nla/2d_chaos_range.png){: .center-image }


# Two Dimensional - Discrete

skNLA also has functions to deal with discrete images. For example:

{% highlight python linenos %}
X = data.voronoiMatrix(percent=.01)
{% endhighlight %}

![xmap_lib_len](/assets/nla/2d_voronoi.png){: .center-image }

Next calculate the mutual information like we did above.

{% highlight python linenos %}
E = nlf.embed(X)
mi = E.mutual_information_spatial(50)
{% endhighlight %}

![xmap_lib_len](/assets/nla/2d_voronoi_mi.png){: .center-image }

As we can see above. It looks like the first minimum is around 10.

{% highlight python linenos %}
lag = (8,8)
emb = (2,2)
predict = 8
X,y = E.embed_vectors_2d(lag,emb,predict,percent=.02)
{% endhighlight %}

Next we train on the first 75% of the data and initiate the object.

{% highlight python linenos %}
#split it into training and testing sets
train_len = int(.75*len(X))
Xtrain = X[0:train_len]
ytrain = y[0:train_len]
Xtest = X[train_len:]
ytest = y[train_len:]

max_nn = .1 * len(Xtrain) # test out to a maximum of 10% of possible NN
weights = 'distance' #use a distance weighting for the near neighbors
NLF = nlf.NonLinDiscrete(max_nn,weights) # initiate the class
{% endhighlight %}

Next we fit the data and calculate the distances.

{% highlight python linenos %}
NLF.fit(Xtrain,ytrain) #fit the training data
NLF.dist_calc(Xtest) #calculate the distance all the near neighbors
{% endhighlight %}

Finally, it needs to be scored.

{% highlight python linenos %}
nn_range = np.arange(1,max_nn,100,dtype='int')
preds = NLF.predict_range(nn_range)
s_range = NLF.score_range(ytest)
{% endhighlight %}

![xmap_lib_len](/assets/nla/2d_voronoi_mi.png){: .center-image }

***
<br>

**1. Calculate mutual information for both time series to find the appropriate lag value.**

Mutual information is used as a way to jump far enough in time that new information about the system can be gained. A similar idea is calculating the autocorrelation. Systems that don't change much from one time step to the next would have higher autocorrelation and thus a larger lag value would be necessary to gain new information about the system. It turns out that using mutual information over autocorrelation allows for better predictions to be made [CITATION].

***

![xmap_changingB](/assets/ccm/lorenz_mutual_info.png){: .center-image }
*Figure:* The image above shows the mutual information for the $$x$$ values of the lorenz time series. We can see a minimum around 16.

***

<br>

**2. Determine the embedding dimension by finding which gives the highest prediction skill.**

Ideally you want to find the best embedding dimension for a specific time series. A good rule of thumb is to use an embedding dimension of three as your first shot. After the initial analysis, you can tweak this hyperparameter until you achieve the best prediction skill.

Alternatively, you can use a [false near neighbor][fnn] test when the reconstructed attractor is fully "unfolded". This functionality is not in skCCM currently, but will be added in the future.

***
![embedding gif](/assets/ccm/embedding.gif){: .center-image }
*Figure:* An example of an embedding dimension of three and a lag of two.

***

<br>



**3. Split each embedded time series into a training set and testing set.**

This protects against highly autocorrelated time series. For example, random walk time series can seem like they are coupled if they are not split into a training set and testing set.

***
![train split](/assets/ccm/train_test_split.png){: .center-image }
*Figure:* Splitting an embedded time series into a training set and a testing set.

***

<br>

**5. Calculate the distance from each test sample to each training sample**

At this point, you will have these four embedded time series:

1. X1tr
2. X1te
3. X2tr
4. X2te

The distance is calculated from every sample in X1te to every sample in X1tr. The same is then done for X2tr and X2te. The distances are then sorted and the closest $$k$$ indices are kept to make a prediction in the next step. $$k$$ is the embedding dimension plus 1. So if your embedding dimension was three, then the amount of near neighbors used to make a prediction will be four.


**6. Use the near neighbor time indices from $$X_1$$ to make a prediction about $$X_2$$**

The next step is to use the near neighbor indices and weights to make a prediction about the other time series. The indices that were found by calculating the distance from every sample in X1te to every sample in X1tr, are used on X2tr to make a prediction about X2te. This seems a little counterintuitive, but it is expected that if one time series influences the other, the system being forced should be in a similar state when the system doing the forcing is in a certain configuration.

INSERT THOUGHT EXPERIMENT

***
![weight switching](/assets/ccm/switching_weights.png){: .center-image }
*Figure:* An example of switching the indices. Notice the distances and indices have the same number of samples as the testing set, but an extra dimension. This is because you need $$K+1$$ near neighbors in order to surround a point.

***
<br>

**7. Repeat the prediction for multiple library lengths**

The hope is we see convergence as the library length is increased. By increasing the library length, the density of the rebuilt attractor is increasing. As that attractor becomes more and more populated, better predictions should be able to be made.

**8. Finally, evaluate the predictions**

The way the predictions are evaluated in the paper is by using the [$$R^2$$][r2] (coefficient of determination) value between the predictions and the actual value. This is done for all the predictions at multiple library lengths. If the predictions for $$X_1$$ are better than $$X_2$$ than it is said that $$X_1$$ influences $$X_2$$.



# Caveats

- Simple attractors can fool this technique (sine waves)
- Can't be used on non-steady state time series.
- Lorenz equation doesn't work?

***

# API

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



### CLASS skNLF.embed( X )

*Embed a 1D, or 2D array*

  - X : series, 2d array, or 3d array to be embedded

#### METHODS

**mutual_information(self,max_lag)**

*Calculates the mutual information between the an unshifted time series and a shifted time series. Utilizes scikit-learn's implementation of the mutual information found in sklearn.metrics.*

- max_lag : integer
  maximum amount to shift the time series

		Returns
		-------
		m_score : 1-D array
			mutual information at between the unshifted time series and the
			shifted time series
		"""

self.X = X






<table class="mbtablestyle">
  <thead>
  </thead>
  <tbody>
    <tr style="text-align: left top;">
      <th>Parameters</th>
      <td>
        <ol>
      		max_nn : int
      			Maximum number of near neighbors to use
      		weights : string
      			-'uniform' : uniform weighting
      			-'distance' : weighted as 1/distance
        </ol>
      </td>

    </tr>
    <tr style="text-align: left;">
      <th>__init__</th>
      <td>

        Fit the training data for ccm. Amount of near neighbors is set to be
        embedding dimension plus one. Creates seperate near neighbor regressors for X1 and X2 independently. Also Calculates the distances to each sample.

        - X1 : embedded time series of shape (num_samps,embed_dim)
        - X2 : embedded time series of shape (num_samps,embed_dim)
        - near_neighs : number of near neighbors to use

      </td>

    </tr>
    <tr style="text-align: left;">
      <th>__init__</th>
      <td>
      <p>
        Fit the training data for ccm. Amount of near neighbors is set to be
        embedding dimension plus one. Creates seperate near neighbor regressors for X1 and X2 independently. Also Calculates the distances to each sample.
        </p>
        <ul>
        <li> X1 : embedded time series of shape (num_samps,embed_dim)</li>
        <li> X2 : embedded time series of shape (num_samps,embed_dim)</li>
        <li>near_neighs : number of near neighbors to use</li>
        </ul>

      </td>

    </tr>
  </tbody>
</table>

<br>
--
<br>

| Item | Description |
| --- | --- |
| item1 |Fit the training data for ccm. Amount of near neighbors is set to be embedding dimension plus one. Creates separate near neighbor regressors for X1 and X2 independently. Also Calculates the distances to each sample.|
| item2 | item2 description |
{:.mbtablestyle}



[wiki-nla]: https://www.wikiwand.com/en/Nonlinear_functional_analysis
[vid-1]: https://www.youtube.com/watch?v=fevurdpiRYg
[vid-2]: https://www.youtube.com/watch?v=rs3gYeZeJcw
[vid-3]: https://www.youtube.com/watch?v=iSttQwb-_5Y
[sug-talk]: https://www.youtube.com/watch?v=uhONGgfx8Do
[paper]: http://science.sciencemag.org/content/338/6106/496
[skccm]:https://github.com/NickC1/skCCM
[r2]: https://www.wikiwand.com/en/Coefficient_of_determination
[fnn]: http://www.mpipks-dresden.mpg.de/~tisean/TISEAN_2.1/docs/chaospaper/node9.html

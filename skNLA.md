---
layout: page
title: skNLA
permalink: /skNLA/
---

**Scikit Nonlinear Analysis**

Scikit Nonlinear Analysis (nla) can be used as a way to forecast time series, spatio-temporal images, and even discrete spatial arangements. More importantly, skNLA can provide insight into the underlying dynamics of a system. For a more complete background, I suggest checking out the [Nonlinear Analysis by Kantz][nla-book]. For a brief overview, the wikipedia article on [nonlinear analysis][wiki-nla] is a good start. Additionally, [Dr. Sugihara's lab][sugihara-lab] has produced some good summary videos of the topic:

1. [Time Series and Dynamic Manifolds][vid-1]
2. [Reconstructed Shadow Manifold][vid-2]

**Package**

skNLA attempts to mimic the ease and style of scikit-learn's api.

**Installation**

`pip install skNLA`

***
<br>

# Simple Example



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

The next step is to calculate the mutual information of the time series so that we can appropriately determine the lag value for the embedding. The first minimum in the [mutual information][mutual-info-wiki] can be thought of as jumping far enough away that there is new information gained. A more useful thought construct might be to think of it as the first minimum in the autocorrelation. Mutual information, however, has proved to be more useful in [appropriately picking the lag][emory-site]. The mutual information calculation can be done using the `embed` class provided by skNLA.

{% highlight python linenos %}
import skNLA as nla

E = nla.embed(X) #initiate the class

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

weights = 'distance' #use a distance weighting for the near neighbors
NLA = nla.NonLin(weights) # initiate the nonlinear forecasting class

{% endhighlight %}

The next step is to then fit the data and calculate the distance from the training set to the testing set.

{% highlight python linenos %}

nn_range = np.arange(1,max_nn,10,dtype='int')
preds = NLA.predict_range(nn_range,Xtest)

score = NLA.score(ytest) #score
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
E = nla.embed(X)
rmi,cmi,rmut,cmut = E.mutual_information_spatial(30)
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
NLF = nla.NonLin(max_nn,weights) # initiate the nonlinear forecasting class
{% endhighlight %}

Next we fit the model, calculate the distances from the training set to the testing set, and finally make some predictions.

{% highlight python linenos %}
NLA.fit(Xtrain,ytrain) #fit the training data

NLA.dist_calc(Xtest) #calculate the distance all the near neighbors

nn_range = np.arange(1,max_nn,10,dtype='int')
preds = NLA.predict_range(nn_range)
{% endhighlight %}

Next, we score the predictions and visualize in a contour plot.

{% highlight python linenos %}
s_range = NLA.score_range(ytest)
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
E = nla.embed(X)
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
NLA = nla.NonLinDiscrete(max_nn,weights) # initiate the class
{% endhighlight %}

Next we fit the data and calculate the distances.

{% highlight python linenos %}
NLF.fit(Xtrain,ytrain) #fit the training data
NLF.dist_calc(Xtest) #calculate the distance all the near neighbors
{% endhighlight %}

Finally, it needs to be scored.

{% highlight python linenos %}
nn_range = np.arange(1,max_nn,100,dtype='int')
preds = NLA.predict_range(nn_range)
s_range = NLA.score_range(ytest)
{% endhighlight %}

![xmap_lib_len](/assets/nla/2d_voronoi_mi.png){: .center-image }

***
<br>


***

# API

### class NonLin(weights):

*PARAMETERS*

- weights : string
	- 'uniform' : uniform weighting
	- 'distance' : weighted as 1/distance

*METHODS*

**fit(Xtrain, ytrain)**

- Xtrain : array (nsamples,nfeatures)
	- Training feature data
- ytrain : array (nsamples,ntargets)
	- Training target data



**dist_calc(Xtest)**

Calculates the distance from the testing set to the training
set.

- Xtest : array (nsamples, nfeatures)
	- test features

**predict(nn_list, Xtest)**

Make a prediction for the given values of near neighbors

- nn_list : list
	- Values of NN to test
- Xtest : array (nsamples, nfeatures)
	- test features

*RETURNS*

- ypred : list; len(nn_list)
	- A list containing the predictions for each nn value


**predict_individual(nn_list, Xtest)**

Instead of averaging near neighbors, make a prediction for each neighbor.

- nn_list : list
	- Values of NN to test
- Xtest : array (nsamples, nfeatures)
	- test features

*RETURNS*

- ypred : list; len(nn_list)
	- A list containing the predictions for each nn value


**score(ytest, how='score')**

Scores the predictions for the numerous values of near neighbors.

- ytest : 2d array (nsamps, ntargets)
	- test data containing the targets
- how : string
	- how to score the predictions
		- 'score' : see scikit-learn's score function
		- 'corrcoef' : correlation coefficient





[nla-book]: https://www.amazon.com/Nonlinear-Time-Analysis-Holger-Kantz/dp/0521529026/ref=sr_1_1?s=books&ie=UTF8&qid=1475599671&sr=1-1&keywords=nonlinear+time+series+analysis
[sugihara-lab]: http://deepeco.ucsd.edu/
[wiki-nla]: https://www.wikiwand.com/en/Nonlinear_functional_analysis
[vid-1]: https://www.youtube.com/watch?v=fevurdpiRYg
[vid-2]: https://www.youtube.com/watch?v=rs3gYeZeJcw
[vid-3]: https://www.youtube.com/watch?v=iSttQwb-_5Y
[lorenz-wiki]: https://www.wikiwand.com/en/Lorenz_system
[chaos-wiki]: https://www.wikiwand.com/en/Chaos_theory
[mutual-info-wiki]: https://www.wikiwand.com/en/Mutual_information
[emory-site]: http://www.physics.emory.edu/faculty/weeks//research/tseries3.html

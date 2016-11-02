---
layout: page
title: skNLA
permalink: /skNLA/
---

**[Scikit Nonlinear Analysis][sknla-github]**

Scikit Nonlinear Analysis (nla) can be used as a way to forecast time series, spatio-temporal 2D arrays, and even discrete spatial arrangements. More importantly, skNLA can provide insight into the underlying dynamics of a system. For a more complete background, I suggest checking out [Nonlinear Analysis by Kantz][nla-book] as well as [Practical implementation of nonlinear time series methods: The TISEAN package][practical-nla]. This package reproduces some of the [tisean package][tisean] in pure python. For a brief overview, the wikipedia article on [nonlinear analysis][wiki-nla] is a good start. Additionally, [Dr. Sugihara's lab][sugihara-lab] has produced some good summary videos of the topic:

1. [Time Series and Dynamic Manifolds][vid-1]
2. [Reconstructed Shadow Manifold][vid-2]


**Installation**

`pip install skNLA`

***
<br>

# Quick Example

In order to illustrate how this package works, we start with the example in the videos above. The [lorenz system][lorenz-wiki] takes the form of :

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

The next step is to calculate the mutual information between the time series and the shifted time series. This determines the lag value for the embedding. The first minimum in the [mutual information][mutual-info-wiki] can be thought of as jumping far enough away that there is new information gained. A more useful thought construct might be to think of it as the first minimum in the autocorrelation. Mutual information, however, is better than autocorrelation for [picking the lag value][emory-site]. The mutual information calculation can be done using the `embed` class provided by skNLA.

{% highlight python linenos %}
import skNLA as nla

E = nla.Embed(X) #initiate the class

max_lag = 100
mi = E.mutual_information(max_lag)
{% endhighlight %}

![mutual_info](/assets/nla/lorenz_mutual_info.png){: .center-image }

The first minimum of the mutual information is at lag=18. This is the lag that will be used to rebuild a shadow manifold. This is done by the `embed_vectors_1d` method. A longer discussion about embedding dimension (how the value for `embed` is chosen) is found in the next section.

{% highlight python linenos %}
lag = 18
embed = 3
predict = 36 #predicting out to double to lag
X,y = E.embed_vectors_1d(lag,embed,predict)
{% endhighlight %}

![x_embedded](/assets/nla/embedded_lorenz.png){: .center-image }

The plot above is showing only `X[:,0]` and `X[:,1]`. This embedding preserves the geometric features of the original attractor.

Now that we embed the time series, all that is left to do is check the forecast skill as a function of near neighbors. First we split it into a training set and testing set. Additionally, we will initiate the class.

{% highlight python linenos %}
#split it into training and testing sets
train_len = int(.75*len(X))
Xtrain = X[0:train_len]
ytrain = y[0:train_len]
Xtest = X[train_len:]
ytest = y[train_len:]

weights = 'distance' #use a distance weighting for the near neighbors
NLA = nla.Regression(weights) # initiate the nonlinear forecasting class

{% endhighlight %}

Next, we need to fit the training data (rebuild the a shadow manifold) and make predictions for the test set.

{% highlight python linenos %}
NLA.fit(Xtrain, ytrain) #fit the data (rebuilding the attractor)

nn_list = np.arange(1,max_nn,10,dtype='int')
ypred = NLA.predict(Xtest, nn_list)

score = NLA.score(ytest) #score
{% endhighlight %}

![xmap_lib_len](/assets/nla/lorenz_forecast_range.png){: .center-image }

As can be seen from the image above, the highest forecast skill is located at low numbers of near neighbors and low forecast distances. In order to view the actual predictions for different numbers of near neighbors we can do the following:

{% highlight python linenos %}
fig,axes = plt.subplots(4,figsize=(10,5),sharex=True,sharey=True)
ax = axes.ravel()

ax[0].plot(ytest[:,35],alpha=.5)
ax[0].plot(ypred[0][:,35])
ax[0].set_ylabel('NN : ' + str(nn_list[0]))

ax[1].plot(ytest[:,35],alpha=.5)
ax[1].plot(ypred[24][:,35])
ax[1].set_ylabel('NN : ' + str(nn_list[24]))

ax[2].plot(ytest[:,35],alpha=.5)
ax[2].plot(ypred[49][:,35])
ax[2].set_ylabel('NN : ' + str(nn_list[49]))


ax[3].plot(ytest[:,35],alpha=.5)
ax[3].plot(ypred[99][:,35])
ax[3].set_ylabel('NN : ' + str(nn_list[99]))

sns.despine()
{% endhighlight %}



![xmap_lib_len](/assets/nla/lorenz_weighted_predictions.png){: .center-image }

As expected, the forecast accuracy decreases as more and more near neighbors are averaged together to make a prediction.

Additionally, instead of averaging near neighbors, it is possible to look at the forecast skill of each near neighbor. This is visualized against the average distance to that point. This is computed as:

{% highlight python linenos %}
NLA.fit(Xtrain, ytrain) #fit the data (rebuilding the attractor)

nn_list = np.arange(1,max_nn,10,dtype='int')
preds = NLA.predict_individual(Xtest, nn_list)

score = NLA.score(ytest) #score
{% endhighlight %}


![xmap_lib_len](/assets/nla/lorenz_score_individual.png){: .center-image }

Likewise, we can look at the actual forecast made by the algorithm and compare it to the actual evolution of the time series.


![xmap_lib_len](/assets/nla/lorenz_individual_predictions.png){: .center-image }

As we can see, by not averaging the near neighbors, the forecast skill decreases and the actual forecast made becomes quite noisy, This is because we are no grabbing points that are not nearby in the space to make predictions. This should intuitively do worse than picking nearby regions.




***
***
***
<br>

# Embedding

Embedding the time series, spatio-temporal series, or a spatial pattern is required before attempting to forecast or understand the dynamics of the system. I would suggest reading [this][emory-site] to understand which lag and embedding dimension is appropriate.

As a quick recap, the lag is picked as the first minimum in the mutual information and the embedding dimension is picked using a false near neighbors test. In practice, however, it is acceptable to use the embedding that gives the highest forecast skill. Through experimentation, an embedding dimension of 3 is a good value to begin with and an embedding dimension of (2,3) for 2d systems.

**1D Embedding**

An example of a 1D embedding is shown in the gif below. It shows a lag of 2, an embedding dimension of 3 and a forecast distance of 2. Setting the problem up this way allows us to use powerful near neighbor libraries such as the one implemented in scikit-learn.

![xmap_lib_len](/assets/nla/1d_embedding.gif){: .center-image }

This is the same thing as rebuilding the attractor and seeing where the point traveled to next. This just makes our lives a little easier.

Using this package, this would be represented as:

{% highlight python linenos %}
E = nla.Embed(X)

lag = 2
embed = 3
predict = 2
X,y = E.embed_vectors_1d(lag, emb, predict)
{% endhighlight %}


More examples of 1d embeddings are shown below. E is the embedding dimension, L is the lag, and F is the prediction distance.

![xmap_lib_len](/assets/nla/1d_embedding_examples.png){: .center-image }

<br>

**2D Embedding**

An example of a 2D embedding is shown in the gif below. It shows a lag of 2 in both the rows and columns, an embedding dimension of two down the rows, and an embedding dimension of three across the columns.

![xmap_lib_len](/assets/nla/2d_embedding.gif){: .center-image }

This would be implemented in code as:

{% highlight python linenos %}
E = nla.Embed(X)

lag = (2,2)
emb = (2,3)
predict = 2
X,y = E.embed_vectors_2d(lag, emb, predict)
{% endhighlight %}

More examples of 2d embeddings are shown below. L is the lag, E is the embedding dimension, and F is the prediction distance.

![xmap_lib_len](/assets/nla/2d_embedding_examples.png){: .center-image }


# Near Neighbors

At the heart of nonlinear analysis is the [k-nearest neighbors algorithm][knn-wiki]. In fact, the package uses scikit-learn's [nearest neighbor implementation][scikit-knn] for efficient calculation of distances and to retrieve the indices of the nearest neighbors. It is a good idea to understand the k-nearest neighbor algorithm before interpreting what this package implements.

For the regression case, we will look at a zoomed in version of the lorenz system that was discussed above. The red dots are the actual points that make up the blue line and the green box is the point that we want to forecast. The trajectory is clockwise.

![xmap_lib_len](/assets/nla/zoom_embedded_lorenz.png){: .center-image }

In this section of the lorrenz attractor, we can see that the red points closest to the green box all follow the same trajectory. If we wanted to forecast this green box, we could grab the closest red point and see where that ends up. We would then say that this is where the green box will end up.

Grabbing more points, however, might prove to be useful since our box lies between a couple of the points. It might be better to average the trajectories of, for example, the three nearest points to make a forecast than just taking the closest one.

It is also possible to imagine that at some point grabbing more and more near neighbors will be detrimental to the forecast as the points that we will be grabbing will have wildly different trajectories. For example, grabbing all the points in this subsection of space will show a trajectory to the right which would be a terrible forecast for the green box.

Additionally, we could also think about adding noise to this system as shown in the plot below.

![xmap_lib_len](/assets/nla/zoom_embedded_lorenz_noise.png){: .center-image }

Now it might be useful to grab more points as the trajectories are no longer smooth. Additionally the trajectories are no longer perfectly deterministic. There is an added level of stochasticity which will lower the forecast skill.

# Evaluation

The next step is to examine the forecast skill. This is done by comparing the actual trajectories to the forecasted trajectories. We will see different patterns in the forecast skill depending if the system is deterministic or noisy. Consider the three systems below.

![xmap_lib_len](/assets/nla/chaos_rand_noise.png){: .center-image }

The top is the [logistic map][logistic-map-wiki]. It is a classic chaotic system. The second is a sine wave with a little bit of noise added. The bottom is white noise. After calculating near neighbors, calculating the forecast and forecast skill, the following plot is produced.

![xmap_lib_len](/assets/nla/forecast_skill_chaos_periodic_noise.png){: .center-image }


Both the logistic map and periodic map $$R^2$$ values fall off as the distance away in the phase space is increased. The sine wave, however, has almost a perfect forecast skill. The plot above is a little different from what was shown in the quick example above. Here we are looking at the forecast skill (y-axis) plotted against the average distance to a particular near neighbor (x-axis). To clarify, the first point on the plots above is the average distance to the first near neighbor for all the points in the testing set. For example, if there were 3 samples in our testing set and the first near neighbor to those points had distances [1.3, 4.5, 2.7] respectively. We would say that the average distance for the first near neighbor is:

$$\frac{.18 + .45 + .27}{3} = .30$$

This would be plotted against the $R^2$ calculated for those three points.

For these three different series, three different trends are apparent. The first is the initial value of the forecast skill. The logistic map and sine wave both have high forecast skills at low distances in the phase space. The white noise, however, has a forecast skill of zero for the first near neighbor. This is to be expected as forecasting a truly noisy system is impossible.

The difference between the sine wave and the Logistic map is that the forecast skill does not dramatically fall off as a function of distance, nor as a function of prediction distance. The $R^2$ value stays high out to a distance of 0.2.




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
E = nla.Embed(X)
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


weights = 'distance' #use a distance weighting for the near neighbors
NLA = nla.Regression(weights) # initiate the nonlinear forecasting class
{% endhighlight %}

Next we fit the model, calculate the distances from the training set to the testing set, and finally make some predictions.

{% highlight python linenos %}
NLA.fit(Xtrain,ytrain) #fit the training data

nn_list = np.arange(1,max_nn,10,dtype='int')
preds = NLA.predict(Xtest,nn_list)
{% endhighlight %}

Next, we score the predictions and visualize in a contour plot.

{% highlight python linenos %}
s_range = NLA.score(ytest)
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
E = nla.Embed(X)
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

weights = 'distance' #use a distance weighting for the near neighbors
NLA = nla.Classification(weights) # initiate the class
{% endhighlight %}

Next we fit the data and calculate the distances.

{% highlight python linenos %}
NLA.fit(Xtrain,ytrain) #fit the training data
{% endhighlight %}

Finally, it needs to be scored.

{% highlight python linenos %}
nn_list = np.arange(1,200,2,dtype='int')
preds = NLA.predict(Xtest, nn_list)
s_range = NLA.score(ytest)
{% endhighlight %}



***

# API

### class Regression(weights):

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


**predict(Xtest, nn_list)**

Make a prediction for the given values of near neighbors

- nn_list : list
	- Values of NN to test
- Xtest : array (nsamples, nfeatures)
	- test features

*RETURNS*

- ypred : list; len(nn_list)
	- A list containing the predictions for each nn value


**predict_individual(Xtest, nn_list)**

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

**dist_stats(nn_list)**
Returns the mean and the standard deviation of the distances for the given nn_list.

*RETURNS*
- mean : 1d array
	- The mean distances for each near neighbors
- std : 1d array
	- The standard devation of the distances for each near neighbor


### class Classification(weights):

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


**predict(Xtest, nn_list)**

Make a prediction for the given values of near neighbors

- nn_list : list
	- Values of NN to test
- Xtest : array (nsamples, nfeatures)
	- test features

*RETURNS*

- ypred : list; len(nn_list)
	- A list containing the predictions for each nn value


**predict_individual(Xtest, nn_list)**

Instead of averaging near neighbors, make a prediction for each neighbor.

- Xtest : array (nsamples, nfeatures)
	- test features
- nn_list : list
	- Values of NN to test

*RETURNS*

- ypred : list; len(nn_list)
	- A list containing the predictions for each nn value


**score(ytest, how='tau')**

Scores the predictions for the numerous values of near neighbors.

- ytest : 2d array (nsamps, ntargets)
	- test data containing the targets
- how : string
	- how to score the predictions
		- 'score' : see scikit-learn's score function
		- 'corrcoef' : correlation coefficient

**dist_stats(nn_list)**
Returns the mean and the standard deviation of the distances for the given nn_list.

*RETURNS*
- mean : 1d array
	- The mean distances for each near neighbors
- std : 1d array
	- The standard devation of the distances for each near neighbor







[practical-nla]: http://scitation.aip.org/content/aip/journal/chaos/9/2/10.1063/1.166424
[tisean]: http://www.mpipks-dresden.mpg.de/~tisean/
[sknla-github]: https://github.com/NickC1/skNLA
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
[knn-wiki]: https://www.wikiwand.com/en/K-nearest_neighbors_algorithm
[scikit-knn]: http://scikit-learn.org/stable/modules/neighbors.html
[logistic-map-wiki]: https://www.wikiwand.com/en/Logistic_map

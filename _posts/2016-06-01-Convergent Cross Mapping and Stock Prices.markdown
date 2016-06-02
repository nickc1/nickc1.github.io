---
layout: post
title:  "Convergent Cross Mapping for Stocks"
date:   2016-06-01 08:00:00 -0500
categories: python,convergent cross mapping,skccm
---


This post explores the use of convergent cross mapping (ccm) as a way to distinguish causation from time series. The full paper, [Detecting Causality in Complex Ecosystems][paper] by Sugihara et al. is an interesting read and I reccomend reading it. Additionally there are a couple of good videos by his lab group:

1. [Time Series and Dynamic Manifolds][vid-1]
2. [Reconstructed Shadow Manifold][vid-2]
3. [Convergent Cross Mapping][vid-3]

If you are interested, there is a [full talk][sug-talk] by Dr. Sugihara that extends some of these ideas into other domains.

Anyway, I decided to reproduce the results from this paper and try to apply the technique to some new time series. This resulted in a package I call [skCCM][skccm]. skCCM tries to follow the ease and style of scikit-learn while implementing convergent cross mapping.

First let's make sure that the package works correctly. Let's test the package with the coupled logistic map from the paper:

$$ X(t+1) = X(t)[r_x - r_x X(t) - \beta_{x,y}Y(t)]$$

$$Y(t+1) = Y(t)[r_y - r_y Y(t) - \beta_{y,x}X(t)]$$

I've included a function in skCCM to reproduce these time series. All that needs to be done is to import it. For example,

{% highlight python linenos %}
import skCCM.data as data

rx1 = 3.72 #determines chaotic behavior of the x1 series
rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = 0.2 #Influence of x1 on x2
b21 = 0.01 #Influence of x2 on x1
ts_length = 1000
x1,x2 = data.coupled_logistic(rx1,rx2,b12,b21,ts_length)
{% endhighlight %}

This produces the standard logistic map as seen in the figure below where the top plot is `x1` and the bottom is `x2`.

![coupled_logistic](/assets/ccm/coupled_logistic.png){: .center-image }

The next step is to calculate the mutual information of the time series so that we can appropriately determine the lag value for the embedding. This can be done using the `embed` class provided by skCCM.

{% highlight python linenos %}
import skCCM.skCCM as ccm

em_x1 = ccm.embed(x1) #initiate the class
em_x2 = ccm.embed(x2)

mi1 = em_x1.mutual_information(10) #call the embed method
mi2 = em_x2.mutual_information(10)
{% endhighlight %}

The top plot below is `mi1` and the bottom is `mi2`
![mutual_info](/assets/ccm/mutual_info.png){: .center-image }

As is seen above, the mutual information is continually decreasing, so a lag of one is sufficient to rebuild a shadow manifold (or a Poincaré section).

{% highlight python linenos %}
lag = 1
embed = 2
X1 = em_x1.embed_vectors_1d(lag,embed)
X2 = em_x2.embed_vectors_1d(lag,embed)
{% endhighlight %}

![x_embedded](/assets/ccm/x_embedded.png){: .center-image }

Now that we have embedded the time series, all that is left to do is check the forecast skill as a function of library length.

{% highlight python linenos %}
lib_lens = np.arange(10,ts_length,ts_length/20)
neighbors = 3
sc1, sc2 = CCM.predict_causation_lib_len(X1,X2,lib_lens,neighbors)
{% endhighlight %}

![xmap_lib_len](/assets/ccm/xmap_lib_len.png){: .center-image }

As can be seen from the image above, `x1` has a higher forecast skill. Another way to view this is that information about `x1` is present in the `x2` time series. This leads to better forecasts for `x1` using `x2`'s reconstructed manifold.

To make sure that this algorithm is robust we test a range of $$\beta$$ values similar to how the original paper. The results below show the difference between `sc1` and `sc2`.

![xmap_changingB](/assets/ccm/xmap_changingB.png){: .center-image }



# Checking Apple and Microsoft Stocks

Common knowledge says that Apple and Microsoft compete in the space of consumer electronics. For this reason we will check to see if one influences the other. For this experiment we will use pandas which makes this all very simple.

First, if you have not already, you will need to install [pandas_datareader][pandas-datareader]. You can simply `conda install pandas-datareader` or `pip install pandas-datareader`. This provides nice clean data frames from stock data.

{% highlight python linenos %}
from pandas_datareader import data, wb

import datetime

start = datetime.datetime(1800, 1, 1)

end = datetime.datetime(2020, 1, 27)

apple = data.DataReader("AAPL", 'yahoo', start, end)
micro = data.DataReader("MSFT",'yahoo', start, end)
{% endhighlight %}

This results in dataframes as follows for Apple and Microsoft respectively.

<table align="center" border="0" class="dataframe" cellpadding="4">
	<caption>Apple</caption>
	<thead>
    <tr style="text-align: center;">
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
			<th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: center;">
      <th>1980-12-12</th>
      <td>28.75</td>
      <td>28.87</td>
      <td>28.75</td>
      <td>28.75</td>
      <td>117258400</td>
			<td>0.431</td>
    </tr>
    <tr style="text-align: center;">
      <th>1980-12-15</th>
      <td>27.37</td>
      <td>27.37</td>
      <td>27.25</td>
      <td>27.25</td>
      <td>43971200</td>
			<td>0.408</td>
    </tr>
  </tbody>
</table>


<table align="center" border="0" class="dataframe" cellpadding="4">
	<caption>Microsoft</caption>
	<thead>
    <tr style="text-align: center;">
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
			<th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: center;">
      <th>1986-03-13</th>
      <td>25.49</td>
      <td>29.24</td>
      <td>25.49</td>
      <td>27.99</td>
      <td>1031788800</td>
			<td>0.066</td>
    </tr>
    <tr style="text-align: center;">
      <th>1986-03-14</th>
      <td>27.99</td>
      <td>29.49</td>
      <td>27.99</td>
      <td>28.99</td>
      <td>308160000</td>
			<td>0.069</td>
    </tr>
  </tbody>
</table>

Next, it is important to line up the time indices in order for ccm to work. This is taken care for us in the coupled logistic map, but for real world data this can be a little messier. Luckily, pandas fixes all of this by simply using one inner join.

{% highlight python linenos %}
combo = apple.join(micro,how='inner',rsuffix=' m')
{% endhighlight %}

This creates a new data frame (dropping the volume, high, and low columns) that looks like:


<table align="center" border="0" class="dataframe" cellpadding="4">
	<thead>
    <tr style="text-align: center;">
      <th>Date</th>
      <th>Open</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Open m</th>
      <th>Close m</th>
			<th>Adj Close m</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: center;">
      <th>1986-03-13</th>
      <td>24.75</td>
      <td>24.75</td>
      <td>0.37</td>
      <td>25.49</td>
      <td>27.99</td>
			<td>0.066</td>
    </tr>
    <tr style="text-align: center;">
      <th>1986-03-14</th>
      <td>24.75</td>
      <td>26.12</td>
      <td>0.391</td>
      <td>27.99</td>
      <td>28.99</td>
			<td>0.069</td>
    </tr>
  </tbody>
</table>


Now that all the dates are lined up, lets do some analysis. For this part, we are going to do something a little different. We are going to check to see if the forcing between the two time series changes after the ipod is released. The first ipod was released on October 23, 2001. This also splits the data almost perfectly in half (1986-2001 and 2001-2016). This is also just a couple lines of code (I love pandas).

{% highlight python linenos %}
before = combo.loc['1986-03-13':'2001-10-23']
after = combo.loc['2001-10-23':]
{% endhighlight %}

![split_data](/assets/ccm/split_data.png){: .center-image }

First lets calculate the mutual information so that we can properly set the lag value.

{% highlight python linenos %}
em_app_before = ccm.embed(before['Adj Close'].values) #initiate the class
em_mst_before = ccm.embed(before['Adj Close m'].values)

em_app_after = ccm.embed(after['Adj Close'].values) #initiate the class
em_mst_after = ccm.embed(after['Adj Close m'].values)

mi_app_after = em_app_after.mutual_information(1000) #calculate mi
mi_mst_after = em_mst_after.mutual_information(1000)

mi_app_before = em_app_before.mutual_information(1000) #calculate mi
mi_mst_before = em_mst_before.mutual_information(1000)
{% endhighlight %}

![mutual_info_stocks](/assets/ccm/mutual_info_stocks.png){: .center-image }


Now lets properly embed them.

{% highlight python linenos %}
lag = 180
embed = 10

X1_before = em_app_before.embed_vectors_1d(lag,embed)
X2_before = em_mst_before.embed_vectors_1d(lag,embed)

X1_after = em_app_after.embed_vectors_1d(lag,embed)
X2_after = em_mst_after.embed_vectors_1d(lag,embed)
{% endhighlight %}

Now that they have been embedded, lets calculate the ccm.

{% highlight python linenos %}
CCM = ccm.ccm()

lib_lens = np.arange(20,2000,2000/100)
sx1_before, sx2_before = CCM.predict_causation_lib_len(
												X1_before,X2_before,lib_lens,11)
sx1_after, sx2_after = CCM.predict_causation_lib_len(
												X1_after, X2_after, lib_lens,11)

{% endhighlight %}


![mutual_info_stocks](/assets/ccm/stock_ccm_results.png){: .center-image }

Interesting! It looks like Microsoft is forcing Apple from 1986-2001 and Apple is forcing Microsoft from 2001-2016. This goes with our intuition about the competition between Apple and Microsoft.

# Fake Data

Let's see if we can generate some fake time series and see if we can get similar results.


{% highlight python linenos %}
def brown_noise_gen(ts_length,seed):
	x1 = np.zeros(ts_length)
	x2 = np.zeros(ts_length)
	np.random.seed(seed)
	x1[0] = np.random.randn()
	x2[0] = np.random.randn()
	for ii in range(ts_length-1):
		x1[ii+1]  = np.random.randn() + x1[ii]
		x2[ii+1] = np.random.randn() + x2[ii]

	return x1,x2

{% endhighlight %}

This was 10000 unique time series were generated. We then looked at the difference between x1's score and x2's score. The histogram is below.

![mutual_info_stocks](/assets/ccm/ccm_hist.png){: .center-image }

Let's look at the top eight and their corresponding time series.

![mutual_info_stocks](/assets/ccm/top_ccm_with_series.png){: .center-image }


The top results look similar to the results we saw for Apple and Microsoft. Although the result is interesting, I am not sure if I believe it 100%. Would need some additional testing.





[vid-1]: https://www.youtube.com/watch?v=fevurdpiRYg
[vid-2]: https://www.youtube.com/watch?v=rs3gYeZeJcw
[vid-3]: https://www.youtube.com/watch?v=iSttQwb-_5Y
[sug-talk]: https://www.youtube.com/watch?v=uhONGgfx8Do
[paper]: http://science.sciencemag.org/content/338/6106/496
[skccm]:https://github.com/NickC1/skCCM
[pandas-datareader]: https://github.com/pydata/pandas-datareader

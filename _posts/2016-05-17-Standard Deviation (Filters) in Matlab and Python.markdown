---
layout: post
title:  "Standard Deviation (Filters) in Matlab and Python"
date:   2016-05-17 08:00:00 -0500
categories: python, Matlab
---


Recently, I was porting some code from Matlab to python when I came across an interesting bit of information. The default standard deviation in Matlab and python do not return the same value. I found this out after messing with python's implementation of a standard deviation filter for half an hour. I thought maybe python's implementation was incorrect. Turn's out they are both correct.

Matlab defaults to the **population** standard deviation:

$$ s_{pop} = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \overline{x})^2} $$

{% highlight matlab linenos %}
x = [1,2,3,4,5];
std(x)

ans =
  1.5811
{% endhighlight %}

While numpy defaults to the **sample** standard deviation:

$$ s_{samp} = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \overline{x})^2} $$

{% highlight python linenos %}
import numpy as np
x = [0,1,2,3,4]
np.std(x)

>>1.4142
{% endhighlight %}

**Lesson Learned:** Always make sure to read to documentation thoroughly.

## Filters

Anyway, my goal was to implement a 2D standard deviation filter that was the same as the [matlab version][matlab-std-filt]. For example the std filter in Matlab returns the following:

{% highlight matlab linenos %}
x = 0:15;
x = reshape(x,4,4)'

x =
  0   1   2   3
  4   5   6   7
  8   9   10  11
  12  13  14  15

stdfilt(x,ones(3,3))

ans =

  2.0616  2.1794  2.1794  2.0616
  3.5000  3.5707  3.5707  3.5000
  3.5000  3.5707  3.5707  3.5000
  2.0616  2.1794  2.1794  2.0616
{% endhighlight %}

Note that on line `2` I transpose the matrix. This way the reshape function will act the same in Matlab as it does in python.

Python does not have a built in std filter, but they do have a [generic filter][generic-filter] that is capable of implementing a standard deviation filter.

{% highlight python linenos %}
from scipy.ndimage.filters import generic_filter

x = np.arange(16).reshape(4,4).astype('float')
x_filt = generic_filter(x, np.std, size=3)
print(x_filt)

[[ 1.9436  2.0548  2.0548  1.9436]
 [ 3.2998  3.3665  3.3665  3.2998]
 [ 3.2998  3.3665  3.3665  3.2998]
 [ 1.9436  2.0548  2.0548  1.9436]]

print(x_filt*np.sqrt(9./8))

[[ 2.0616  2.1794  2.1794  2.0616]
 [ 3.5000  3.5707  3.5707  3.5000]
 [ 3.5000  3.5707  3.5707  3.5000]
 [ 2.0616  2.1794  2.1794  2.0616]]
{% endhighlight %}

Notice that `x_filt*np.sqrt(9./8)` produces the same output as the Matlab function. More formally,

$$s_{pop} = \sqrt{\frac{N}{N-1}}s_{samp}$$

While experimenting with the python function, however, I noticed it was quite slow. I should say brutally slow. Searching around I found a [stackoverflow post][stack-overflow] asking about performance. A user, nneonneo, suggests a much quicker implementation that you can see on the linked stackoverflow post. After fiddling with his code a little bit, I was able to perfectly reproduce the results from scipy's `generic_filter`. The code that replicates scipy's function is:

{% highlight python linenos %}
from scipy.ndimage.filters import uniform_filter

def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)

x = np.arange(16).reshape(4,4).astype('float')
window_stdev(x,3)

[[ 1.9436  2.0548  2.0548  1.9436]
 [ 3.2998  3.3665  3.3665  3.2998]
 [ 3.2998  3.3665  3.3665  3.2998]
 [ 1.9436  2.0548  2.0548  1.9436]]

{% endhighlight %}

As you can see, it returns the same values as the python filter. If you want to return the same values as the Matlab function, all you have to do is multiply the returned value by $$\frac{window\_size^2}{window\_size^2-1}$$ which is what was done above since the window size was 3.

Just to prove how much faster this implementation is than the generic filter, here are some benchmarks on different size arrays.

<table align="center" border="0" class="dataframe" cellpadding="4">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>Generic</th>
      <th>Uniform</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: center;">
      <th>20 x 20</th>
      <td>0.0157</td>
      <td>0.0004</td>
    </tr>
    <tr style="text-align: center;">
      <th>500 x 500</th>
      <td>7.5842</td>
      <td>0.0114</td>
    </tr>
    <tr style="text-align: center;">
      <th>1,000 x 1,000</th>
      <td>30.0421</td>
      <td>0.0581</td>
    </tr>
  </tbody>
</table>

Finally, as a sanity check to make sure they both output the same results on randomly sized matrices:

{% highlight python linenos %}
x = np.random.rand(765,478)

quick_filt = window_stdev(x,5)
slow_filt = generic_filter(x,np.std,size=5)

np.sum(quick_filt-slow_filt)

-4.9917e-12
{% endhighlight %}

And there we are. A quick implementation of a standard deviation filter in python that produces the same results as the Matlab version. A big thank you to nneonneo for the original implementation.





[matlab-std-filt]: http://www.mathworks.com/help/images/ref/stdfilt.html?requestedDomain=www.mathworks.com

[generic-filter]: http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.generic_filter.html

[stack-overflow]: http://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows

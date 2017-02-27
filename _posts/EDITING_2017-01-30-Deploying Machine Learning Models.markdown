---
layout: post
title:  "Deploying Models to Amazon"
date:   2017-01-30 08:00:00 -0500
categories: python
---


Problem:

I have a trained machine learning model. I want to let someone else (developers) call it using a Rest API.

First let's explore the options for deploying a model. I came across a [stack overflow question][stack-quesiton] pertaining to R and this [Quora question][quora-python] that was having a similar problem and looking for a solution. In the answers, one user advocates for a couple paid services. These include:

  1. [Domino](domino)
  2. [Azure](azure)
  3. [yhat](yhat)

These all carry significant costs and I am a minimalist at heart. I enjoy the small simple solutions that solve a specific problems instead of solutions that include everything even the kitchen sink. Domino is a great platform. It allows you to simply upload your code in R and it will create a rest API for you. It also allows you to store various versions of your code and even scale up the machine power used to run your code and explore hyper parameters.

I wanted something small. I started to look for these solutions and eventually landed on [Amazon Web Services](aws). The first platform I checked out was [Amazon Machine Learning](amazon-learning). This seemed like a great solution. The procedure for creating a machine learning model is pretty simple:

1. Upload your clean data to Amazon S3
2. Point the Amazon Machine Learning API at the data
3. Train a Logistic Regression Model or Linear Regression Model
4. Create Batch Predictions or Real Time Predictions

Amazon Machine Learning creates a nice API for you to query your results and is very user friendly. The only problem is that you have to have cleaned/transformed data available before you can use it. So if the software engineers have to transform raw data, then it is no good. For example, if you use any non-linear dimensionality reduction, or perform feature engineering from the raw features, then this solution is not appropriate. Additionally if you want to use any other algorithm besides logistic regression or linear regression, this model is also not appropriate. You can not use decision trees, neural networks, or even a k-near neighbor classifier/regressor.

Thus I pressed on.

The next stop was the consideration of building my own rest-api using Flask. I found a [nice blog post](flask-api-post) describing the process. This way I could build exactly what I wanted and launch it to an EC2 instance. This would be the most flexible solution. The drawbacks, however, are I would have to manage an EC2 instance and learn how to build an API. Since this will be handling relatively sensitive information, this also might not be the way to go.


## Amazon Lambda

The next stop was Amazon Lambda. This seemed like a good compromise between the two. No servers to manage. All I have to do is upload my code and we were off and running. I could create an api using amazons built-in API service. I'm sure this would be more secure than anything that I came up with. This is what I am going to explore now.




[scikit-lambda]: https://serverlesscode.com/post/scikitlearn-with-amazon-linux-container/
[cloud-academy]: https://github.com/cloudacademy/sentiment-analysis-aws-lambda
[cloud-academy-youtube]: https://www.youtube.com/watch?v=fdIDn3hr27k
[quora-python]: https://www.quora.com/What-is-the-best-way-to-deploy-a-scikit-learn-machine-learning-model-on-AWS

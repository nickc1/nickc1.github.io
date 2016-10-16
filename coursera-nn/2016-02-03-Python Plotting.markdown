---
layout: post
date:   mm2016-10-03 08:00:00 -0500
categories: machine learning
---

# Why Do we Need Machine learning

#### What is Machine Learning?

- It is very hard to write programs that solve problems like recognizing a three-dimensional object from a novel viewpoint in new lighting
conditions in a cluttered scene.
	- We don’t know what program to write because we don’t know
how its done in our brain.
	- Even if we had a good idea about how to do it, the program might
be horrendously complicated.
- It is hard to write a program to compute the probability that a credit
card transaction is fraudulent.
	- There may not be any rules that are both simple and reliable. We
need to combine a very large number of weak rules.
	- Fraud is a moving target. The program needs to keep changing.

#### The Machine Learning Approach

- Instead of writing a program by hand for each specific task, we collect lots of examples that specify the correct output for a given input.
- A machine learning algorithm then takes these examples and produces a program that does the job.
	- The program produced by the learning algorithm may look very different from a typical hand-written program. It may contain millions of numbers.
	- If we do it right, the program works for new cases as well as the ones we trained it on.
	- If the data changes the program can change too by training on the new data.
- Massive amounts of computation are now cheaper than paying someone to write a task-specific program.

#### Some examples of tasks best solved by learning
- Recognizing patterns:
	+ Objects in real scenes
	+ Facial identities or facial expressions
	+ Spoken words
- Recognizing anomalies:
	+ Unusual sequences of credit card transactions
	+ Unusual patterns of sensor readings in a nuclear power plant
- Prediction:
	+ Future stock prices or currency exchange rates
	+ Which movies will a person like?

#### A standard example of machine learning
- A lot of genetics is done on fruit flies.
	+ They are convenient because they breed fast.
	+ We already know a lot about them.
- The MNIST database of hand-written digits is the the machine learning equivalent of fruit flies.
	+ They are publicly available and we can learn them quite fast in a moderate-sized neural net.
	+ We know a huge amount about how well various machine learning methods do on MNIST.
- We will use MNIST as our standard task.

#### Beyond MNIST: The ImageNet task
- 1000 different object classes in 1.3 million high-resolution training images from the web.
	+ Best system in 2010 competition got 47% error for its first choice and 25% error for its top 5 choices.
- Jitendra Malik (an eminent neural net sceptic) said that this competition is a good test of whether deep neural networks work well for object recognition.
	+ A very deep neural net (Krizhevsky et. al. 2012) gets less that 40% error for its first choice and less than 20% for its top 5 choices (see lecture 5).


#### The Speech Recognition Task

- A speech recognition system has several stages:
	+ Pre-processing: Convert the sound wave into a vector of acoustic coefficients. Extract a new vector about every 10 mille seconds.
	+ The acoustic model: Use a few adjacent vectors of acoustic coefficients to place bets on which part of which phoneme is being spoken.
	+ Decoding: Find the sequence of bets that does the best job of fitting the acoustic data and also fitting a model of the kinds of things people say.
- Deep neural networks pioneered by George Dahl and Abdel-rahman Mohamed are now replacing the previous machine learning method for the acoustic model.

#### Phone recognition on the TIMIT benchmark *(Mohamed, Dahl, & Hinton, 2012)*

- After standard post-processing using a bi-phone model, a deep net with 8 layers gets 20.7% error rate.
- The best previous speaker independent result on TIMIT was 24.4% and this required averaging several models.
- Li Deng (at MSR) realized that this result could change the way speech recognition was done.

#### Reasons to study neural computation
-  To understand how the brain actually works.
	+ Its very big and very complicated and made of stuff that dies when you poke it around. So we need to use computer simulations.
- To understand a style of parallel computation inspired by neurons and their
adaptive connections.
	+ Very different style from sequential computation.
- should be good for things that brains are good at (e.g. vision)
- Should be bad for things that brains are bad at (e.g. 23 x 71)
- To solve practical problems by using novel learning algorithms inspired by the brain (this course)
	+ Learning algorithms can be very useful even if they are not how the brain actually works.


#### A typical cortical neuron
- Gross physical structure:
	+ There is one axon that branches
	+ There is a dendritic tree that collects input from other neurons.
- Axons typically contact dendritic trees at synapses
	+ A spike of activity in the axon causes charge to be injected into the post-synaptic neuron.
- Spike generation:
	+ There is an axon hillock that generates outgoing spikes whenever enough charge has flowed in at synapses to depolarize the cell membrane.
- When a spike of activity travels along an axon and arrives at a synapse it causes vesicles of transmitter chemical to be released.
	+ There are several kinds of transmitter.
- The transmitter molecules diffuse across the synaptic cleft and bind to receptor molecules in the membrane of the post-synaptic neuron thus changing their shape.
	+ This opens up holes that allow specific ions in or out.

#### How synapses adapt
- The effectiveness of the synapse can be changed:
	+ vary the number of vesicles of transmitter.
	+ vary the number of receptor molecules.
- Synapses are slow, but they have advantages over RAM
	+ They are very small and very low-power.
	+ They adapt using locally available signals
- But what rules do they use to decide how to change?
- Each neuron receives inputs from other neurons
	+ A few neurons also connect to receptors.
	+ Cortical neurons use spikes to communicate.
- The effect of each input line on the neuron is controlled by a synaptic weight
	+ The weights can be positive or negative.
- The synaptic weights adapt so that the whole network learns to perform useful computations
	+ Recognizing objects, understanding language, making plans, controlling the body.
- You have about neurons each with about weights.
	+ A huge number of weights can affect the computation in a very short time. Much better bandwidth than a workstation.

#### Modularity and the brain
- Different bits of the cortex do different things.
	+ Local damage to the brain has specific effects.
- Specific tasks increase the blood flow to specific regions.
	+ But cortex looks pretty much the same all over.
- Early brain damage makes functions relocate.
- Cortex is made of general purpose stuff that has the ability to turn into special purpose hardware in response to experience.
	+ This gives rapid parallel computation plus flexibility.
	+ Conventional computers get flexibility by having stored sequential programs, but this requires very

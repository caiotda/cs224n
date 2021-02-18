## Language models



A Language model computes de **probability of occurence** of a number of words in a particular sequence. If the sequence of words is {w1, w2, w3, ..., wm} its probability is P(w1, w2, w3, ..., wm). 

A common use of language models is next word suggestion present on most mobile OS. If you're typing the phrase "I love to pet my ...." the system should be able to predict that the next word will be something like "cat" or "dog".



Because we're dealing with text, the probability that a given word occurs next depends heavily on its previous words. But we're usually not interested in all words in our text that come before the target word. Therefore, **we're only concerned with the n previous words.** Thats why we model a language model as:
$$
P(w_1, w_2, w_3, \dots, w_m) = \prod_{i=1}^m P(w_i \ | \ w_{i-n}, \dots, w_{i-1})
$$

 ### n-gram language models

But how do we calculate that probability? One way to do so is to assemble a **n-gram language model**. Recall from statistics that the following holds:


$$
P(a|b, c, d) = \frac{P(a, b, c, d)}{P(b, c, d)}
$$
So, in n-gram language models we'll look at a window of size n. Lets look of an exemple for a 3-gram model:
$$
P(w_3|w_1, w_2) = \frac{count(w_1, w_2, w_3)}{count(w_1, w_2)}
$$
We calculate each probability just by counting every time we see the set of this words **in this exact order**. 



Noticed that we have fixed our context. But how can we choose a good value of n? For an exemple, look at the following sentence:



"After hearing a meow, zoey pet her...."



If we use a context window of size 3 (that is, a 3-gram model), we would be evaluating:


$$
P(?|her, pet) = \frac{P(?, het, pet)}{P(her, pet)}
$$
Where "?" is a word that maximizes this probability. Its reasonable to assume that its a toss up between cat or dog: its very likely that our corpus of text contain equal amounts of information about cats and dogs. But if we were to use a 4-gram model, our program would consider the word "meow" and the chances of the "?" word to be "cat" would skyrocket. 



But at the same time, what if we chose a n that was far too large?



N-gram models have two main issues: **sparsity and storage**:

* Sparsity problem: notice the numerator in the above equation. What if P(?, her, pet) is 0? That is, what if "<something> her pet" never occurs in our corpus? Its absurd to assume that a word <something> should never be suggested (our corpus could just be incomplete, or the word never appeared on the test set). We could fix this via **smoothing**: we just add a small $\delta$ to every probability so that every word **could** be suggested

  Another issue is due to the denominator: What if "her pet" never occurs on our corpus? Then this probability could never be calculated. If count(wordA, wordB) is 0, we just look at count(wordA). This strategy is called **backoff**.

* **storage problem**: We need to store the count for all n-grams we saw in the corpus. As n grows, the model size increases as well.



### Window-based Neural  Language model



Another approach would be to just used a regular feed forward neural network.

We would represent the window of size n as n wordvectors that are fed to the neural network. The neural network would then output a probability distribution that would give the most likely word to come next. Unfortunately, we're still left with a storage problem, as the size n of the window would impact heavily our model size and predictive power.



## Recurrent Neural Networks



This architecture takes the cake.

Unlike the two previous models, RNNs don't require a fixed number of words to condition a model, as they can take **any number of words**, even being capable of conditioning the model on **all** previous words.

### RNN architecture

A RNN consists of the same building blocks as a usual feed forward neural network:

* input nodes
* Output nodes
* Hidden nodes: responsible for intermediate computation

The main difference is that RNN operate in a cyclical fashion: instead of receiving a fixed amount and starting the computation, RNNs periodically **combine inputs with hidden layer inputs to compute its hidden output**. The output then can be used as an input again for the hidden layer with new input or passed along as an output:

![](/home/caio/Downloads/reducao(1).png)

If we have the time series representation in mind it will be easier to explain how this architecture works. Notice that our neural network doesn't have several hidden nodes: **each node is a time step. The neural networks consists of a single neuron**



Now, a bit of notation:

* $X_t$: input at time step t; Because we're dealing with words, X will be a word vector
* $\hat{y}_t$: output at time step t
* $H_t$: output of the hidden node at timestep t
* $W_x$ weight matrix associated with the inputs
* $W_t$: weight matrix associated with the hidden node output
* $W_y$: weight matrix associated with the neural network final output

The output of a neuron is always passed through a non linear function before being used as an input on the next time step (we'll be using sigmoid for notation). The neural network output also uses a non linear function, but to map the output to a probability distribution (we'll use softmax). We then can model a RNN mathematically as:
$$
H_t = \sigma(W_t \times H_{t-1} + W_x \times X_t)
\\
\hat y _t = softmax(W_y \times H_t)
$$
So, at each time step t we perform a linear operation on he input at time step t with the output of a neuron in time step t-1.

Notice that, for all of the Neural network, **there is only one matrix associated with the input, one matrix associated with the neural network output and one matrix associated with the neuron output**. Regardless of how many words we're using the predict the next word, our model size won't grow.

Although our example had a output at each time step, it's perfectly reasonable to use a single output at the end of the RNN, or customize it however you want it. In a couple of lectures we'll see some different RNN architectures.

### Advantages

* RNNs are flexible regarding input size
* Our model size doesn't increase for longer input sequence lengths
* In theory, at time step t we can access information from many steps back

### disadvantages

* Slow computation: due to it's sequential architecture, we can't use parallel code to optimize computation with RNNs
* In practice its difficult to reach information from many steps back due to a problem called **vanishing gradient** and **exploding gradient**. We'll talk about it in a couple of sections.

### Computing loss with RNNs

We'll be using **cross entropy loss** for RNNs. The loss is calculated as:
$$
j^{(t)}(\theta)  = -\sum_{j=1}^{|V|}y_{t,j}\times log(\hat y_{t, j})
\\
j(\theta) = \frac{1}{T}\sum _{t=1}^{T}j^{(t)}(\theta)
$$
Where |V| is the size of our vocabulary and T is the size of the corpus.



Because this computation is slow, we usually consider just a sentence.

We can use J(theta) to estimate how good is our model by looking at its **perplexity**:
$$
perplexity = 2^j
$$
This is a measure of confuse. Lower values imply more confidence in predicting the next word in the sequence.

### Training RNNs


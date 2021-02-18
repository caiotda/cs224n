# Neural translation



We've dealt with a couple of tasks previously that required only one output: the Named Entity Recognition (NER) of a given word, the most likely next word, sentiment analysis in a phrase, etc.



In NLP, we also deal with tasks that require a sequence of outputs:

* **Translation**: Take a sequence in one language and output the same sequence in another language
* **Conversation**: take a statement or question as input and output the answer to it.
* Summarization: receive a long text and try to output a shorter text (a summary)

A good deep learning model to approach these kind of tasks is the Sequence to sequence (**seq2seq**) architecture. It is the standart for neural translation in the present.



### Historical approaches

Before NMT, translation systems were composed by two separated probabilistic systems:

1. A translation model: responsible to determine which word is more likely to be the translation of a word in the input phrase
2. A language model: given a sentence/word, it should tell us how likely it was

Therefore, the first system would translate the sentence, and the language model would adjust the translation to be somewhat reasonable in the new language. So if we were to translate the sentence "Io sono un ragazzo" from Italian to English, we would need a translation model from Italian to English and a English language model.



One particular issue that this system had was that it failed completely to capture differences in ordering between languages (the negating term might be used differently in different languages), so the translation was a bit clunky.

### Seq2Seq basics

A seqseq is madeup from two rnn's:

1. An encoder: it takes the input and outputs a fixed-size context vector
2. A decoder: a language model. it uses the context vector as a seed to generate text.

For this reason, Seq2Seq models are referenced as encoder-decoder models. 



It's important to say that a seq2seq model is a single system, unlike the aforementioned historical approaches above: both the encoder and decoder are trained together, constituting a single system. 

### Architecture

* encoder: The encoder is madeup of a LSTM that reads each input, one at a time. Because it's difficult to compress all of the input information into a single context vector, the encoder RNN will actually be made-up of two layers of LSTMS. The output of each layer is the input of the next layer. Fun fact: the encoder will usually encode the input sentence in reverse: this way the first thing the decoder looks at is the start of the input sequence (think about this operation as a stack, and the decoder will be unstacking items);

* Decoder: The decoder will be sort of a conditioned language model. The context vector will be the input to the RNN and it will dictate what the decoder must generate. Furthermore, the decoder needs to be aware of two things:

  1. The input context
  2. The words that it is generating.

  Therefore, we'll keep the stacked architecture from the encoder, this way we can keep track of the generated words and the input context vectore.

  So the whole stack of LSTMs will output a softmax whose argmax will represent a single word. Besides outputting the single word, it will pass this through the next timestep to the next stack of LSTMs and repeat until the end of the sequence.

### Bidirectional RNNs and limitations

One limitation that our current architecture has is that it doesn't considers information from future words, only previous words. Using bidirectional RNNs could help with that.



This is done by traversing the sentence in two directions: forwards and backwards. Then, instead of having the output t as a cell state, we have a concatenation as output: [output forwards, output backwards]. The same goes for the hidden state of the rnn, which will be a concatenation of two different hidden states.



## Attention mechanism

A problem with the previous architecture was that the context vector was outputted only at the end of our LSTM. What this means is that the context valued more short term information than long term information (even with a LSTM RNN, the gradients can vanish). So we're interested in a context that captures the information of the whole sentence to be translated; furthermore, it would be cool to designate weights to important terms. For example, if we want to translate "the boy played with the ball", the terms "boy", "played" and "ball" are the core of the sentence, so it's very important to get them right.



Attention is a **neural technique** that solves this issue. **At each step of the decoder, we'll use a direct connection to the encoder to focus on a particular part of the original sentence**.



For each step of the decoder, we'll try to draw connections to each input token. This connection will be put through a softmax function, which will **output a probability distribution**: so that we understand what term from the original sentence relates to the current term



So, the hidden state s_i at timestep i can be written as:
$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$
Where y_i-1 is the outputted word at timestep i-1 and c_i is the context vector **with the information that is more important at timestep i**. This is where attention comes in. To calculate ci, we'll need to calculate e_i and alpha_i:



e_i is just a score:
$$
e_i = a(s_{i-1}, h_j)
$$
Where "a" is any function that maps values to R. h_j is the hidden state at timestep j in the **original sentence, therefore, in the encoder section**.



This will calculate a sequence of scalars e_0, e_1, e_2, ..., e_n. We'll normalize this scores using a softmax and obtain $\alpha_i$  = $(\alpha_0, \alpha_1, \alpha_2, \dots, \alpha_n)$. This is called the **attention vector**:
$$
\alpha_{i,j} = \frac{exp(e_{i, j})}{\sum_{k=1}^{n} exp(e_{i, k})}
$$
Finally, the context vector is just the weighted average of the hidden vectors from the original sentence:
$$
c_i = \sum_{j=1}^{n} \alpha_{i,j} * h_j
$$
Because of attention, NMT becomes powerful to translate longer sentences.
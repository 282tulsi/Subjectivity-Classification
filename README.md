Subjectivity detection with Bi-RNNs
===================================

Installation
------------

```bash
virtualenv --python=/usr/bin/python3 .env 
source .env/bin/activate 
pip install -r requirements.txt
wget http://nlulite.com/download/glove ./data/word_embeddings/
```

Running a simple example
------------------------

Pipe a text file from the command line: 
```bash
python -m subjectivity.classification-subjective1 < data/input.txt
```

The output will look like the following:
```text
OBJECTIVE SENTENCES
<list of objective sentences in the text>

SUBJECTIVE SENTENCES
<list of subjective sentences in the text>
```


\newpage

Network Structure
-----------------

The network structure is as in Fig. 1

![The network structure](./docs/images/BiGRU.png){height=50%}

A text is divided into sentences. Each sentence is tokenized into words, and 
each word vector is given as an input to the network.

For the word embeddings I have used the 50-dim Glove vectors trained on 
[[Wikipedia 2014](https://nlp.stanford.edu/projects/glove/)].

Each word embeddings goes through a dense layer. The result is then fed to a bi-directional GRU, 
effectively composed by a forward GRU and a backward one. 
The last state of the forward GRU and first state of the backward GRU are 
concatenated and fed into one last dense layer, which is finally 
projected with a softmax into a binary vector.

This last vector represents the predicted class: [1, 0] predicts a subjective sentence, 
[0, 1] an objective one.

|Layer | dimension  |
|------|:----------:|
|Word embeddings     | 50 |
|Dense layer     | 25 |
|GRU memory     | 100 |
|GRU stacks     | 1 |
|Final hidden dense layer     | 200 |
|Output  size    | 2 |
|Minibatch size    | 10 |
|Dropout rate    | 0.3 |

\newpage



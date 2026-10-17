# FableForge: Autoregressive Next-Word Text Generation with Recurrent Neural Networks

**Foundations of Generative AI (ICT4403), Assignment 2**

| | |
|---|---|
| Student | Qusai |
| Registration No. | 230953480 |
| Developed system | FableForge, a word-level autoregressive text generator on Indian Fairy Tales, comparing Simple RNN, Bidirectional RNN, LSTM and GRU |
| Dataset | Indian Fairy Tales, compiled by Joseph Jacobs (1892), Project Gutenberg eBook #7128 |
| Dataset source | https://www.gutenberg.org/ebooks/7128 |

## 1. Problem statement and objective

Build an autoregressive next-word text-generation system and train it with four recurrent architectures
(Simple RNN, Bidirectional RNN, LSTM and GRU) under comparable experimental conditions. The objectives
are to:

1. describe and preprocess a textual dataset and build the input-target sequences required for
   autoregressive modelling, then implement and train all four models;
2. evaluate and compare the models quantitatively (loss, perplexity, top-1 and top-5 prediction
   accuracy, parameter count, training time) and qualitatively (coherence, diversity, contextual
   consistency, repetition);
3. generate text from the same seed sequences at three sampling temperatures and analyse the effect of
   temperature on coherence, diversity and repetition, then justify which model gives the most effective
   autoregressive generation for this dataset.

The deliverable is a self-contained Jupyter notebook (FableForge_textgen.ipynb) that reproduces every
number, table, plot and generated sample reported here.

## 2. Dataset description and source

Dataset: Indian Fairy Tales, compiled by Joseph Jacobs, illustrated by John Dickson Batten
(Project Gutenberg eBook #7128, public domain).
Source link: https://www.gutenberg.org/ebooks/7128

It is a collection of 29 folk and fairy tales drawn from Indian oral and written traditions (Jataka
tales, Panchatantra-style fables and regional folk stories), written in consistent English narrative
prose. This makes it a good fit for word-level language modelling: a single style, recurring characters
(kings, princes, rajas, jackals, cranes) and formulaic phrasing.

Corpus statistics after cleaning:

| Property | Value |
|---|---|
| Characters | 304,759 |
| Total words (tokens) | 58,104 |
| Unique words | 4,332 |
| Vocabulary kept (built from the training segment) | 3,000 plus the two special tokens pad and unk |
| Token coverage of kept vocabulary | 97.7 percent |

## 3. Preprocessing

The plain-text edition is downloaded from Project Gutenberg, then:

* the Project Gutenberg licence header and footer are removed;
* the front matter (preface, contents, list of illustrations) is removed;
* the scholarly Notes and References appendix at the end is removed. This matters because otherwise
  roughly the last fifth of the file is academic prose with a very different distribution from the
  tales, which distorts any held-out evaluation;
* the text is lower-cased and tokenised on alphabetic runs, keeping apostrophes;
* the token stream is split 80/10/10 into contiguous train, validation and test segments, so no context
  window crosses a segment boundary;
* the vocabulary of the 3,000 most frequent words is built from the training segment only, and all other
  words map to the unknown token, with index 0 reserved for padding.

Input and target construction. A sliding window of 10 consecutive words is the input and the
immediately following word is the target. A stride of 8 controls the number of sequences. This yields
5,808 training, 726 validation and 726 test next-word targets.

## 4. Models implemented

All four models share the identical interface, an embedding layer followed by a recurrent layer and a
dense softmax head, so the comparison is like for like. Only the recurrent layer changes.

| Model | Recurrent layer | Direction |
|---|---|---|
| SimpleRNN | nn.RNN | unidirectional |
| BiRNN | nn.RNN | bidirectional |
| LSTM | nn.LSTM | unidirectional |
| GRU | nn.GRU | unidirectional |

Note on the Bidirectional RNN. A bidirectional layer reads the input sequence in both directions, so
when it predicts the next word it can exploit future words inside the input window. The next-word task
is therefore technically non-causal for the BiRNN, and a real autoregressive generator cannot use it. It
is included here as required by the brief, and this caveat is carried into the conclusion.

Shared configuration, identical for every model: embedding dim 64, hidden size 128, dropout 0.2, Adam
optimiser, learning rate 0.001, batch size 512, 8 epochs, cross-entropy loss, the same train, validation
and test split, and a fixed random seed of 42. Checkpoints were selected by validation loss and each
model was evaluated once on the test split.

## 5. Results

All numbers below are from the executed notebook with a fixed seed of 42.

| Model | Test loss | Perplexity (lower better) | Top-1 (higher better) | Top-5 (higher better) | Parameters | Training time (s) |
|---|---|---|---|---|---|---|
| Bidirectional RNN | 5.6019 | 270.94 | 10.88 percent | 28.51 percent | 1,012,664 | 33 |
| Simple RNN | 5.7572 | 316.47 | 7.58 percent | 21.07 percent | 603,832 | 20 |
| GRU | 5.7851 | 325.41 | 8.40 percent | 26.03 percent | 653,496 | 34 |
| LSTM | 5.7961 | 329.02 | 7.58 percent | 19.01 percent | 678,328 | 43 |
| Unigram baseline | 7.9022 | 2703.23 | 7.58 percent | n/a | n/a | n/a |

Perplexity is the exponential of the test loss. Top-1 and top-5 are next-word accuracy on the 726-target
test split. The context-free unigram baseline, which always predicts the most frequent training word, is
included for reference; its top-1 accuracy equals the frequency of the word "the" in the test targets.

Generated samples, identical seed "the king said to", 35 words, three temperatures:

SimpleRNN

* T = 0.5: the king said to the me he to the the the was to he unk very in the the the the and and the to the the the and a the and and the to the the my very
* T = 1.0: the king said to gangazara at to to his which hand saying in former hands see only but his she fear unk and the the none replied them and have prince to when as only has the with down
* T = 1.5: the king said to wait gates matter sure him returned most if which king's the can them after it already happily fond livelihood fire you yet plain touch still said how last neither followed has skilled agreed now i

BiRNN

* T = 0.5: the king said to unk the whole they of the young and then and and he unk the tiger the young to and the king of the raja and i am and the in the young and he was
* T = 1.0: the king said to her turn had by unk the tiger them could him husband dip world and will he she busy he let out here she unk of me the down the they so and if ring wicked
* T = 1.5: the king said to him the begging everything matter your kill whom tree misdeeds for rohu replied boy road song sciences him in nevertheless grain which i i softened as would is has he given he spite when because

LSTM

* T = 0.5: the king said to the and a the and he and the the of and unk he the he the then the the the the him killed the and he was and the the to unk and the a
* T = 1.0: the king said to brute in court you flowers woman set unk communication over you given his thick and back is the of unk they you very you you answered the window speak did go seek he the of
* T = 1.5: the king said to bone that were before chanced their at yawning the counsel slaves sorrow what the trying army opened it bird assembled hearing news queens' brothers a sat of daughter could he princess you cricket measures he

GRU

* T = 0.5: the king said to be hear the but the to you then one and the king and he and a the the see the of the unk the the king and their of the you to you the and
* T = 1.0: the king said to horse you morning when then daughter lotus the slaves was by the as came the princess held and sarkap as seven him he of the you these beak pitcher rasalu and was she in have
* T = 1.5: the king said to kindly brother living grip with the dropped grave rats chose single affair farmer another saying on jackal little inhabitants sons laughed could to and put his stable axe sticks wedding my bridle meal then mothers

## 6. Model comparison and evaluation

Quantitative. All four models beat the unigram baseline, whose perplexity is 2703, by a wide margin.
The Bidirectional RNN reaches the lowest test loss of 5.60, the lowest perplexity of 270.94 and the
highest top-1 and top-5 accuracy of 10.88 percent and 28.51 percent. Contrary to the usual expectation
that gating wins, the plain Simple RNN at 316.5 also edges out the LSTM at 329.0 and the GRU at 325.4.
The most likely reason is that on this fixed eight-pass budget the gated models, which carry more
parameters, had not yet converged. The GRU is the more efficient gated model: it uses fewer parameters
than the LSTM, 653k against 678k, has a slightly lower loss and a clearly higher top-5 accuracy of 26.0
percent against 19.0 percent.

Qualitative.

* Coherence. All four models learn the corpus register, and the generated text is grammatical English in
  the fairy-tale idiom, for example "the king", "the prince" and "the raja's".
* Diversity. At low temperature the models are repetitive, and at higher temperature they explore more
  vocabulary, as measured by distinct-2 below.
* Contextual consistency. All models respect the seed and keep the royal court and village subject
  matter, but none sustains a plot over more than a phrase or two. This is the main weakness.
* Repetition. At temperature 0.5 the Simple and Bidirectional RNNs fall into short high-frequency loops
  and occasional unknown-token loops, while the gated models are more varied.

The loss curves in figures/loss_curves.png show all four models improving sharply for the first three
epochs and then slowly, with a widening train and validation gap, which means the models begin to
overfit the small corpus after the initial epochs. The accuracy curves in figures/accuracy_curves.png
show the same shape, with validation accuracy rising quickly and then flattening.

## 7. Sampling temperature analysis

Text is generated one word at a time. The model's softmax distribution over the next word is divided by
a temperature T before sampling. A temperature below 1 sharpens the distribution, which is more coherent
and more repetitive, and a temperature above 1 flattens it, which is more diverse and less coherent.
Three temperatures were used, 0.5, 1.0 and 1.5, with nine continuations per temperature (three seed
sequences and three stochastic draws). Three metrics are averaged over the continuations: distinct-2,
the ratio of unique bigrams to total bigrams, the repeated-trigram rate, which is 1 minus distinct-3, and
the unknown-token rate.

Raw sampling:

| Model | Distinct-2 (0.5 / 1.0 / 1.5) | Repeated-trigram (0.5 / 1.0 / 1.5) | Unknown-token (0.5 / 1.0 / 1.5) |
|---|---|---|---|
| SimpleRNN | 0.678 / 0.997 / 1.000 | 0.118 / 0.000 / 0.000 | 0.029 / 0.014 / 0.003 |
| BiRNN | 0.817 / 0.997 / 1.000 | 0.064 / 0.000 / 0.000 | 0.017 / 0.017 / 0.000 |
| LSTM | 0.646 / 0.994 / 1.000 | 0.134 / 0.000 / 0.000 | 0.009 / 0.020 / 0.014 |
| GRU | 0.850 / 1.000 / 1.000 | 0.024 / 0.000 / 0.000 | 0.031 / 0.017 / 0.003 |

See figures/temperature_diversity.png. The trend is consistent across all models: as temperature rises,
diversity rises and repetition falls.

* Temperature 0.5 is the most fluent and coherent, but also the most repetitive. High-frequency words
  dominate and short phrases repeat.
* Temperature 1.0 is the best trade-off. The text is mostly grammatical and on topic, with markedly
  more varied vocabulary.
* Temperature 1.5 gives the most diverse output, but coherence degrades and the text becomes a sequence
  of loosely related words.

Top-k filtering with k equal to 50. Filtering reads better but can raise repetition. At temperature 0.5
the repeated-trigram rate rises from 0.134 to 0.196 for the LSTM and from 0.064 to 0.133 for the BiRNN.
The key finding is the classic coherence and diversity trade-off controlled by temperature.

## 8. Conclusion: best-performing model

The Bidirectional RNN is the best-performing model on the quantitative measures, with the lowest test
loss of 5.60, the lowest perplexity of 270.94 and the highest top-1 and top-5 accuracy of 10.88 percent
and 28.51 percent. However its advantage is partly not legitimate for autoregressive generation, because
it reads the input window in both directions and can see future words when predicting the next one, so
it cannot generate text in a truly causal, left to right fashion.

For a genuine autoregressive text generator the fairest choice is therefore the GRU. It has the highest
top-1 and top-5 accuracy among the causal models, at 8.40 percent and 26.03 percent, uses the fewest
parameters of the gated units and trains quickly. The LSTM is a close alternative, the BiRNN is reported
as the best predictor but is unsuitable for true generation, and the SimpleRNN is the fastest baseline.

Limitations. A single training seed, a fixed eight-epoch budget, a short 10-word context and a small
test set of 726 targets. With more data, a longer budget and stronger regularisation, the gated models
would be expected to overtake the plain RNN.

## 9. Repository structure

    fableforge-indian-fairytales-rnn/
      FableForge_textgen.ipynb     the full implementation, preprocessing, models, evaluation, generation
      README.md                    this file
      requirements.txt
      data/
        indian_fairy_tales_raw.txt  downloaded automatically on first run
      figures/
        loss_curves.png
        accuracy_curves.png
        model_comparison.png
        temperature_diversity.png
      models/
        SimpleRNN.pt
        BiRNN.pt
        LSTM.pt
        GRU.pt
      results.json                 all metrics, histories and generated samples

## 10. Execution steps and dependencies

Dependencies, see requirements.txt: Python 3.10 or later, torch (CPU is sufficient), numpy, matplotlib,
jupyter and nbformat. No GPU is required.

    python -m venv venv
    source venv/bin/activate        on Windows use venv\Scripts\activate
    pip install -r requirements.txt
    jupyter notebook FableForge_textgen.ipynb
    then choose Cell, then Run All

Running all cells on a modern laptop CPU takes roughly four to six minutes. The notebook downloads the
dataset to data/, trains the four models, writes the plots to figures/, saves the weights to models/ and
dumps all metrics and generated samples to results.json. All results were produced with a fixed random
seed of 42, and small numeric differences may occur across hardware and library versions.

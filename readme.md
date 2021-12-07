# Chat disentanglement with BERT embeddings

## Overview
"Chat disentanglement" or "thread detection" is an NLP task that involves training a computational model to recognize different conversations in a dataset of text. Specifically, the logs of a "chatroom" will have multiple users and multiple conversations, with utterances from each conversation being "asynchronous" or interleaved.

Elsner and Charniak (2008) was an early attempt to make a thread detection model, using data the authors annotated from a Linux tech support IRC channel. Their approach included a binary linear classifier that attempted to predict if two utterances belonged to the same conversation. Then, this trained model was used in a partitioning algorithm to cluster an entire test-set into separate conversations. The algorithmn starts at the beginning of the dataset, and adds new utterances to a cluster if the classifier gives the highest probability prediction to the current utterance and the last one in that cluster. If it predicts negatively for all clusters so far, a new one is created and the current utterance is the first entry.

Since Elsner and Charniak released their data to the public, I wanted to know if a replication of their study, using pre-trained BERT embeddings, would improve the classification and partitioning accuracy. In addition to their "number of shared tokens at unigram probability *i*" features, for the new model I wanted to consider the sum of cosine distances of all tokens at unigram probability *i*. I thought that this would add a bit more semantic information, since two utterances in the same conversation don't necessarily share any tokens at all. However, they're likely to share words in a similar semantic domain, and therefore tokens with higher cosine similarities.

## Results
I trained a baseline model, using a similar feature set to Elsner and Charniak, and a new model, using both the original features and the new BERT-based features. I found a modest improvement - the new model out-performed the baseline on classification accuracy by around three percentage points. Interestingly, it out-performed the baseline on cluster overlap by around nine points, even though the same partitioning algorithm was used. It could be that a small improvement in the classifier leads to more consistent correct decisions while partitioning.

| Model               | Dev accuracy | Partition overlap |
| ------------------- | ------------ | ----------------- |
| Elsner and Charniak | 75.0%        | 51.12%            |
| My baseline         | 77.8%        | 68.3%             |
| With BERT           | 80.2%        | 77.1%             |

The one downside to the new model is the extra time needed for looking up BERT embeddings. Since this kind of model could be deployed "in real time" for use in a chatroom, this extra time in the featurizing stage could make it less practical to run on a CPU than the baseline model.

I made my own baseline model, because it's difficult to compare my results with Elsner and Charniak's. They most likely used different approaches to preprocessing, constructing their models, and splitting up their data. I wouldn't be surprised if they had access to more/different data than what was publicly released.

## To use
The data is available [here.](https://www.asc.ohio-state.edu/elsner.14/resources/chat-manual.html) For this model, I used the files `IRC/dev/linux-dev-0X.annot` and `IRC/pilot/linux-pilot-0X.annot`. To run the models, run the commands `$ python3 thread_detect.py baseline` or `$ python3 thread_detect.py new`.

### Code files
- `thread_detect.py`: Top-level code for running the models.
- `featurize.py`: Classes for preprocessing, featurizing, and embedding lookups.
- `models.py`: Classes for the Classifier and Partitioner models.

### Libraties used
- numpy
- sklearn
- torch
- nltk (word_tokenize function)
- scipy (cosine function)
# pyonion
A minimal Python implementation of the ONe Instance ONly algorithm for text deduplication, based on http://corpus.tools/wiki/Onion. Minimal means that this is intended to provide only the functionality to clean a corpus of text based on similarity between segments of documents, and does not include the most extreme memory management required to handle very large corpora. For example if you have two documents:

1. The cat sat on the mat
2. The cat sat on the rug

We wish to identify these as near duplicates, in a memory efficient way without spending an inordinate amount of time comparing each document to each other document. If your corpus is small (less than 100,000 news article sized documents) than this methodology will be overkill, and slower than simply doing everything in memory, however if your corpus is large (1 million+ documents) then this provides a way to discover duplicated n-grams and remove similar chunks of text with manageable memory requirements.

## Installation
Installation using pip:
```bash
pip install pyonion
```

If you wish to use the Spark implementation (when complete!):
```bash
pip install pyonion[spark]
```

## Quickstart
### Method
The main idea behind this algorithm is that rather than comparing each document with every other document to see how similar they are to create a set of **shingles**, also interchangeably called **n-grams** in many contexts, which appear multiple times in the overall corpus. **Shingles** are sequences of words, parts of complete sentences. For example 'the cat sat on the mat' could be decomposed into shingles of length 3 giving ('the cat sat'), ('cat sat on'), ('sat on the'), ('on the mat'). 

If we choose the shingle length to be sufficiently long that we avoid common phrases, we can then be confident that if we see the same shingle in multiple documents it is likely duplicated text. If a large number of a documents shingles are in the set of duplicated shingles, we define that document as being of low quality, and it can be removed from the corpus, or labelled as a duplicate.

Document cleaning is then a linear time operation, as each document needs only be compared against the set of duplicated shingles.

### Implementation
First create a `CorpusProvider` which will handle documents. The `ListCorpusProvider` is used to deliver documents from an existing list.

```python
from pyonion.remover import ListCorpusProvider

documents = [
    'The cat sat on the large mat',
    'The cat sat on the large rug'
         ]
corpus = ListCorpusProvider(documents)
```

To then discover all 5-grams in the corpus that occur at least twice create a `DuplicateRemover`.
```python
from pyonion.remover import DuplicateRemover, CleaningMode

remover = DuplicateRemover(n_gram=5)
duplicated_ngrams = remover.find_duplicated_ngrams(corpus)
```

This will discover all sequences of words of length 5 that are repeated in the corpus multiple times.

```python
>>> duplicated_ngrams
('The_cat_sat_on_the', 'cat_sat_on_the_large')
```
Use these ngrams to then remove documents that contain a lot of (>20%) duplicated content. Note that the clean text method returns an iterator.
```python
iter_clean_corpus = remover.clean_text(corpus, duplicated_ngrams, threshold=.2, mode=CleaningMode.FIRST)
clean_corpus = [clean_doc for clean_doc in iter_clean_corpus]
```
Note that the second entry has been removed entirely, as it was more than 20% duplicate content, leaving only an empty string. The number alongside each sentence is the resemblance between the original document and 
```python
>>>clean_corpus
[
    ('The cat sat on the large mat', 0.0), 
    ('', 0.6666666666666666)
]
```

# Advanced Usage
### Creating your own corpus generator
The corpus generator passed to the `find_duplicated_ngrams` must be a function that returns an iterable of iterables. Practically this means it should be something that can be looped over that returns lists of lists. 

The simplest way to avoid loading the full dataset is to read chunks of data in as they are required. This does mean that the full dataset is read once per n-gram (i.e. 7-grams require 7 full data reads). The simplest corpus generator uses Pandas to read data in chunks, do a little processing and then yield those chunks.

```python
import pandas as pd
from pyonion.utils import simple_tokenizer


def generate_corpus(data_location, chunksize):
    for df_chunk in pd.read_csv(data_location, usecols=['Verbatim'], chunksize=chunksize):
        df_chunk['Verbatim'].fillna('', inplace=True)
        corpus = df_chunk['Verbatim'].apply(simple_tokenizer)
        yield corpus
```

To use this corpus generator function it is easiest to create a partial function using functools. This function can then be passed as an argument to `find_duplicated_ngrams`.

```python
from functools import partial

corpus_generator = partial(generate_corpus, data_location=data_location, chunksize=100_000)
duplicated_ngrams = find_duplicated_ngrams(corpus_generator, n_gram=5, threshold=2)
```

To then clean the corpus the same generator can be re-used in the `clean_text` function. However in this case to again preserve memory the cleaned text can be written out in chunks.

```python
output_path = 'clean_data.txt'

for clean_chunk in clean_text(corpus_generator,
                              duplicated_ngrams,
                              threshold=.2,
                              mode='oneinstance'):
    with open(output_path, 'w+') as f:
        f.writelines(clean_chunk)
```

### Using hashed n-grams
n-grams are stored internally simply as strings joined using the `join_char` selected - the default being `'_'`. When working with a large corpus the set of n-grams becomes the greatest memory constraint. To reduce the memory usage 64-bit (int) hashes of n-grams are stored, rather than the strings themselves. This dramatically reduces the memory requirements of the process, at the cost of the n-grams themselves being unrecoverable from the hashes and the very small chance of a collision between hashes. To use hashed values set the argument `hash_values` to be `True`.

```python
remover = DuplicateRemover(n_gram=5, hash_values=True)
```
If you try to look at the results of `find_duplicated_ngrams` you'll notice that integers are returned rather than text!





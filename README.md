# pyonion
A minimal Python implementation of the ONe Instance ONly algorithm for text deduplication, based on http://corpus.tools/wiki/Onion. Minimal means that this is intended to provide only the functionality to clean a corpus of text based on similarity between segments of documents, and does not include the most extreme memory management required to handle very large corpora. For example if you have two documents:

1. The cat sat on the mat
2. The cat sat on the rug

We wish to identify these as near duplicates, in a memory efficient way without spending an inordinate amount of time comparing each document to each other document. If your corpus is small (less than 100,000 news article sized documents) than this methodology will be overkill, and slower than simply doing everything in memory, however if your corpus is large (10 million+ documents) then this provides a way to discover duplicated n-grams and remove similar chunks of text with manageable memory requirements.

## Installation
Installation using pip:
```bash
pip install pyonion
```

If you wish to use the Spark implementation:
```bash
pip install pyonion[spark]
```

## Quickstart
Rather than comparing all documents against all other documents and discovering the resemblence between them, a set duplicated n-grams are discovered, where n is sufficiently high that common phrases won't be picked up. A value of 10 does well in my experience.

A `corpus` is an iterable of tokenized documents, for example a list of lists of strings.

```python
corpus = [
    ['The', 'cat', 'sat', 'on', 'the', 'mat'],
    ['The', 'cat', 'sat', 'on', 'the', 'rug']
         ]
```
Discover all 5-grams in the corpus that occur at least twice.
```python
from pyonion import find_duplicated_ngrams

duplicated_ngrams = find_duplicated_ngrams(corpus, n_gram=5, threshold=2)
```

```python
>>>duplicated_grams
['The_cat_sat_on_the']
```
Use these ngrams to then remove documents that contain a lot of duplicated content. 
```python
clean_corpus = clean_text(corpus, duplicated_ngrams, threshold=.2, mode='oneinstance')
```
```python
>>>clean_corpus
[['The', 'cat', 'sat', 'on', 'the', 'mat'],
 ['']]
```
Note that the second entry has been removed entirely, as it was more than 20% duplicate content, leaving only an empty string.

## Advanced Usage
Two top level methods are provided. `find_duplicated_ngrams` is used to discover n-grams that are duplicated within you dataset, and `clean_text` to remove bad documents.

The process is set up to handle large datasets. In the case your documents will not fit into memory `find_duplicated_ngrams` will also accept a generator of chunks of data as its first argument.

### Setting up a corpus generator
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
n-grams are stored internally simply as strings joined using the `join_char` selected - the default being `'_'`. When working with a large corpus the set of n-grams becomes the greatest memory constraint. To reduce the memory usage 64-bit (int) hashes of n-grams are stored, rather than the strings themselves. This dramatically reduces the memory requirements of the process, at the cost of the n-grams themselves being unrecoverable from the hashes and the very small chance of a collision between hashes. To use hashed values after importing the pyonion module do:
```python
import pyonion

pyonion.HASH_VALUES = True
```
If you try to look at the results of `find_duplicated_ngrams` you'll notice that integers are returned, rather than text!

## More info 


## Benchmarking
On a corpus



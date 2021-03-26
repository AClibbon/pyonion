"""
A rough Python implementation of http://corpus.tools/wiki/Onion.

Will be significantly slower, as does not use C++, or the Judy memory arrays, though is suitable for smaller corpora.
"""
import logging
import re
import time
from collections import Counter
from typing import List, Set
from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Don't log by default

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters
# thanks https://stackoverflow.com/questions/53240763/python-how-to-separate-paragraphs-from-text
TOKENIZE_RE = re.compile(r"\W+")
DEFAULT_N_CHUNKS = 128
HASH_VALUES = False  # If true then work using hashed values - extra computation cost but lower memory requirements.


def find_duplicated_ngrams(corpus_generator, n_gram: int, threshold=2, join_char='_') \
        -> Set[str]:
    """
    Efficiently find duplicated n_grams in a corpus of documents.

    :param join_char: Join ngrams with this e.g. [New, York, Times] -> New_York_Times
    :param corpus_generator: A process to generate tokenized text.
    :param n_gram: Length ngram to discover.
    :param threshold: Minimum number of occurrences [default 2]
    :return: A set of strings which represent the discovered ngrams, joined by the concatenation character.
    """
    logger.info("Finding Unigrams")
    duplicated_ngrams = _find_duplicated_unigrams(corpus_generator, threshold)

    for i in range(2, n_gram + 1):
        start_time = time.perf_counter()
        logger.info(f"Finding {i}-grams")
        duplicated_ngrams = _find_duplicated_ngrams(corpus_generator, i, duplicated_ngrams, threshold, join_char)
        logger.info(f"Found {len(duplicated_ngrams)} duplicated {i}-grams in {time.perf_counter() - start_time:.2f}"
                    f" seconds.")
    return duplicated_ngrams


def join_counters(counter1, counter2):
    counter1.update(counter2)
    return counter1


def _find_duplicated_unigrams(corpus_generator, threshold):
    unigram_counts = Counter()

    for i, corpus_chunk in enumerate(corpus_generator()):
        start_time = time.perf_counter()
        chunk_counts = find_unigram_counts(corpus_chunk)
        logger.info(f"Chunk {i}: Found {len(chunk_counts)} unigrams in {time.perf_counter() - start_time:.2f} seconds.")
        unigram_counts.update(chunk_counts)

    if HASH_VALUES:
        duplicated_unigrams = {hash(unigram) for unigram, count in unigram_counts.items() if count >= threshold}
    else:
        duplicated_unigrams = {unigram for unigram, count in unigram_counts.items() if count >= threshold}
    logger.info(f"Found {len(duplicated_unigrams)} unigrams occurring at least {threshold} time(s).")
    print(unigram_counts.most_common(10))

    return duplicated_unigrams


def _find_duplicated_ngrams(corpus_generator, n, nminusonegrams, threshold=2, join_char='_'):
    """
    Finds duplicated n_grams in a memory efficient way

    The process is not computationally, but is memory efficient. n_grams are only counted if both 'halves' of an ngram
    are  present in the nminusonegrams list, for example (New York Times) would only be counted if both (New York) and
    (York Times) were in the set nminusonegrams. This prevents the list of ngrams from becoming staggeringly large.

    :param corpus_generator: A list of tokenised documents
    :param nminusonegrams: A set of n-1 grams to consider. For the return value to be accurate these must appear at
                           least n times in the text.
    :param threshold: Minimum occurrences to count as duplicated
    :return: A set of ngrams that appear at least n times in the corpus.
    """
    ngram_counts = Counter()

    for i, corpus_chunk in enumerate(corpus_generator()):
        start_time = time.perf_counter()
        chunk_counts = find_ngram_counts(corpus_chunk, n=n, check_against=nminusonegrams, join_char=join_char)
        logger.info(f"Chunk {i}: Found {len(chunk_counts)} {n}-grams in "
                    f"{time.perf_counter() - start_time:.2f} seconds.")
        ngram_counts.update(chunk_counts)

    duplicated_ngrams = {ngram for ngram, count in ngram_counts.items() if count >= threshold}
    logger.info(f"Found {len(duplicated_ngrams)} {n}-grams occurring at least {threshold} time(s).")
    return duplicated_ngrams


def find_unigram_counts(corpus):
    """Returns a count of unigrams in the corpus"""
    counter = Counter()
    for doc in corpus:
        counter.update(doc)
    return counter


def find_ngram_counts(corpus, n, check_against, join_char):
    """Returns a count of ngrams in the corpus"""
    counter = Counter()
    for doc in corpus:
        counter.update(get_n_grams_w_checking(doc, n, check_against, join_char))
    return counter


def get_n_grams_w_checking(tokens, n, check_against, join_char):
    assert n > 1, "This won't work for unigrams as no components"

    n_grams = []

    lhs = join_tokens(tokens[0:n - 1], join_char)

    for i in range(len(tokens) - n + 1):
        rhs = join_tokens(tokens[(i + 1):(i + n)], join_char)
        if (lhs in check_against) and (rhs in check_against):
            n_grams.append(join_tokens(tokens[i:(i + n)], join_char))
        lhs = rhs  # Saves calculating next time

    return n_grams


def join_tokens(tokens, join_char):
    if HASH_VALUES:
        return hash(join_char.join(tokens))
    else:
        return join_char.join(tokens)


def get_n_grams(tokens, n, join_char):
    return [join_tokens(tokens[i:i + n], join_char) for i in range(len(tokens) - n + 1)]


def clean_text_in_blocks(text: str, blockizer, tokenizer, duplicated_ngrams: Set[str], threshold: float) -> str:
    """
    Removes low quality blocks from text

    Splits a document into blocks using the provided blockizer (or paragraph splitter), then finds the number of
    duplicated n_grams in that block. If the resemblance between the block and the duplicate set is above the threshold
    that block will be removed. If all text is of low quality will return an empty string.

    :param text: Text to clean
    :param blockizer: Method str -> list(str) to split text into paragraphs/blocks
    :param tokenizer: Method to tokenize text str -> list(str)
    :param duplicated_ngrams: Set of duplicated ngrams
    :param threshold: Remove blocks with resemblance above this
    :return: A string of cleaned text. If all text is poor quality will return an empty string.
    """
    pass


def simple_blockizer(text: str) -> List[str]:
    """Splits text at paragraph markers"""
    return NEWLINES_RE.split(text)


def simple_tokenizer(sentence: str) -> List[str]:
    """Splits text at whitespace markers"""
    return TOKENIZE_RE.split(sentence)

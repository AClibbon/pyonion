"""
A rough Python implementation of http://corpus.tools/wiki/Onion.

Will be significantly slower, as does not use C++, or the Judy memory arrays, though is suitable for smaller corpora.
"""
import logging
import re
import time
from collections import Counter
from typing import List, Set, Iterable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Don't log by default

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters
# thanks https://stackoverflow.com/questions/53240763/python-how-to-separate-paragraphs-from-text
TOKENIZE_RE = re.compile(r"\W+")
DEFAULT_N_CHUNKS = 128
HASH_VALUES = False  # If true then work using hashed values - extra computation cost but lower memory requirements.
JOIN_CHAR = '_'  # Character used to join n-grams  e.g. [New, York, Times] -> New_York_Times


def find_duplicated_ngrams(corpus_generator, n_gram: int, threshold=2) \
        -> Set[str]:
    """
    Efficiently find duplicated n_grams in a corpus of documents.

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
        duplicated_ngrams = _find_duplicated_ngrams(corpus_generator, i, duplicated_ngrams, threshold)
        logger.info(f"Found {len(duplicated_ngrams)} duplicated {i}-grams in {time.perf_counter() - start_time:.2f}"
                    f" seconds.")
    return duplicated_ngrams


def join_counters(counter1, counter2):
    counter1.update(counter2)
    return counter1


def _find_duplicated_unigrams(corpus, threshold):
    """
    Finds unigrams occurring more than N times

    :param corpus: EITHER a function that yields corpus chunks or a corpus of text.
    :param threshold: Return only unigrams occurring more than this many times.
    :return: A set of unigrams
    """
    unigram_counts = Counter()

    if callable(corpus):
        for i, corpus_chunk in enumerate(corpus()):
            start_time = time.perf_counter()
            chunk_counts = find_unigram_counts(corpus_chunk)
            logger.info(f"Chunk {i}: Found {len(chunk_counts)} unigrams "
                        f"in {time.perf_counter() - start_time:.2f} seconds.")
            unigram_counts.update(chunk_counts)
    else:
        unigram_counts = find_unigram_counts(corpus)

    if HASH_VALUES:
        duplicated_unigrams = {hash(unigram) for unigram, count in unigram_counts.items() if count >= threshold}
    else:
        duplicated_unigrams = {unigram for unigram, count in unigram_counts.items() if count >= threshold}
    logger.info(f"Found {len(duplicated_unigrams)} unigrams occurring at least {threshold} time(s).")

    return duplicated_unigrams


def _find_duplicated_ngrams(corpus, n, nminusonegrams, threshold=2):
    """
    Finds duplicated n_grams in a memory efficient way

    The process is not computationally, but is memory efficient. n_grams are only counted if both 'halves' of an ngram
    are  present in the nminusonegrams list, for example (New York Times) would only be counted if both (New York) and
    (York Times) were in the set nminusonegrams. This prevents the list of ngrams from becoming staggeringly large.

    :param corpus: EITHER a function generating lists of tokenised documents or a list of tokenised documents.
    :param nminusonegrams: A set of n-1 grams to consider. For the return value to be accurate these must appear at
                           least n times in the text.
    :param threshold: Minimum occurrences to count as duplicated
    :return: A set of ngrams that appear at least n times in the corpus.
    """
    ngram_counts = Counter()

    if callable(corpus):
        for i, corpus_chunk in enumerate(corpus()):
            start_time = time.perf_counter()
            chunk_counts = find_ngram_counts(corpus_chunk, n=n, check_against=nminusonegrams)
            logger.info(f"Chunk {i}: Found {len(chunk_counts)} {n}-grams in "
                        f"{time.perf_counter() - start_time:.2f} seconds.")
            ngram_counts.update(chunk_counts)
    else:
        ngram_counts = find_ngram_counts(corpus, n=n, check_against=nminusonegrams)

    duplicated_ngrams = {ngram for ngram, count in ngram_counts.items() if count >= threshold}
    logger.info(f"Found {len(duplicated_ngrams)} {n}-grams occurring at least {threshold} time(s).")
    return duplicated_ngrams


def find_unigram_counts(corpus):
    """Returns a count of unigrams in the corpus"""
    counter = Counter()
    for doc in corpus:
        counter.update(doc)
    return counter


def find_ngram_counts(corpus, n, check_against):
    """Returns a count of ngrams in the corpus"""
    counter = Counter()
    for doc in corpus:
        counter.update(get_n_grams_w_checking(doc, n, check_against))
    return counter


def get_n_grams_w_checking(tokens, n, check_against):
    assert n > 1, "This won't work for unigrams as no components"

    n_grams = []

    lhs = join_tokens(tokens[0:n - 1])

    for i in range(len(tokens) - n + 1):
        rhs = join_tokens(tokens[(i + 1):(i + n)])
        if (lhs in check_against) and (rhs in check_against):
            n_grams.append(join_tokens(tokens[i:(i + n)]))
        lhs = rhs  # Saves calculating next time

    return n_grams


def join_tokens(tokens):
    if HASH_VALUES:
        return hash(JOIN_CHAR.join(tokens))
    else:
        return JOIN_CHAR.join(tokens)


def get_n_grams(tokens: List[str], n: int) -> List[str]:
    return [join_tokens(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def calc_resemblance(doc_ngrams: Set, comparison_ngrams: Set):
    """
    Discover ratio of document n_grams present in the target set
    """
    n_ngrams = len(doc_ngrams)
    n_in_both = len(doc_ngrams.intersection(comparison_ngrams))
    return n_in_both / n_ngrams


def clean_text(corpus: Iterable[List[str]], duplicated_ngrams: Set[str], threshold: float, mode: str, n_gram: int)\
        -> List[str]:
    """
    Removes documents with a high ratio of duplicated text

    In first mode keep an running record of duplicated n_grams that have been encountered already.

    :param corpus: An iterable of tokenized documents
    :param mode: Either 'first' or 'all'. Behaviour when encountering duplicated text. If 'first' then will keep the
                 first occurrence encountered, else if set to all will remove all occurrences seen.
    :param duplicated_ngrams: Set of duplicated ngrams
    :param threshold: If resemblance with duplicated text is above this then remove this document.
    :param n_gram: Length of n_gram used to identify duplication
    :return: An iterator of cleaned documents.
    """
    if mode == 'first':
        seen_n_grams = set()

        for document in corpus:
            doc_ngrams = set(get_n_grams(document, n_gram))
            resemblance = calc_resemblance(doc_ngrams, seen_n_grams)
            seen_n_grams.update(doc_ngrams.intersection(duplicated_ngrams))

            if resemblance >= threshold:
                yield ''
            else:
                yield ' '.join(document)

    elif mode == 'all':
        for document in corpus:
            doc_ngrams = set(get_n_grams(document, n_gram))
            resemblance = calc_resemblance(doc_ngrams, duplicated_ngrams)

            if resemblance >= threshold:
                yield ''
            else:
                yield ' '.join(document)
    else:
        raise NotImplementedError(f"Mode {mode} not recognised. Must be either 'first' or 'all'")


def clean_text_in_blocks(corpus: Iterable[str], blockizer, tokenizer, duplicated_ngrams: Set[str], threshold: float,
                         mode: str) -> Iterable[str]:
    """
    Removes low quality blocks from text

    Splits a document into blocks using the provided blockizer (or paragraph splitter), then finds the number of
    duplicated n_grams in that block. If the resemblance between the block and the duplicate set is above the threshold
    that block will be removed. If all text is of low quality will return an empty string.

    :param corpus:
    :param mode: Either 'first' or 'all'. Behaviour when encountering duplicated text. If 'first' then will keep the
                 first occurrence encountered, else if set to all will remove all occurrences seen.
    :param blockizer: Method str -> list(str) to split text into paragraphs/blocks
    :param duplicated_ngrams: Set of duplicated ngrams
    :param threshold: Remove blocks with resemblance above this
    :return: A Iterator over cleaned text. If all blocks of text are poor quality will return an empty string.
    """
    if mode == 'first':
        seen_n_grams = set()

        for document in corpus:
            blocks = blockizer(document)
            for block in blocks:
                tokens = tokenizer(block)
                doc_ngrams = set(get_n_grams(tokens, n_gram))
                resemblance = calc_resemblance(doc_ngrams, seen_n_grams)
                seen_n_grams.update(doc_ngrams.intersection(duplicated_ngrams))

                if resemblance >= threshold:
                    yield ''
                else:
                    yield ' '.join(document)

    elif mode == 'all':
        for document in corpus:
            doc_ngrams = set(get_n_grams(document, n_gram))
            resemblance = calc_resemblance(doc_ngrams, duplicated_ngrams)

            if resemblance >= threshold:
                yield ''
            else:
                yield ' '.join(document)
    else:
        raise NotImplementedError(f"Mode {mode} not recognised. Must be either 'first' or 'all'")


def simple_blockizer(text: str) -> List[str]:
    """Splits text at paragraph markers"""
    return NEWLINES_RE.split(text)


def simple_tokenizer(sentence: str) -> List[str]:
    """Splits text at whitespace markers"""
    return TOKENIZE_RE.split(sentence)

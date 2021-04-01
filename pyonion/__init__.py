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
JOIN_CHAR = '_'  #


class DuplicateRemover:
    def __init__(self, hash_values=False, join_char='_', n_gram=10, duplication_threshold=2):
        """

        :param hash_values: If true then work using hashed values - extra computation cost but lower memory requirements
        :param join_char: Character used to join n-grams  e.g. [New, York, Times] -> New_York_Times
        :param n_gram: Length
        :param duplication_threshold: n-grams occurring more than this many times count as duplicates [default 2]
        """
        self.hash_values = hash_values
        self.join_char = join_char
        self.n_gram = n_gram
        self.threshold = duplication_threshold

    def find_duplicated_ngrams(self, corpus_generator) -> Set[str]:
        """
        Efficiently find duplicated n_grams in a corpus of documents.

        :param corpus_generator: A process to generate tokenized text.
        :return: A set of strings which represent the discovered ngrams, joined by the concatenation character.
        """
        logger.info("Finding Unigrams")
        duplicated_ngrams = self._find_duplicated_unigrams(corpus_generator)

        for i in range(2, self.n_gram + 1):
            start_time = time.perf_counter()
            logger.info(f"Finding {i}-grams")
            duplicated_ngrams = self._find_duplicated_ngrams(corpus_generator, i, duplicated_ngrams)
            logger.info(f"Found {len(duplicated_ngrams)} duplicated {i}-grams in {time.perf_counter() - start_time:.2f}"
                        f" seconds.")
        return duplicated_ngrams

    def _find_duplicated_unigrams(self, corpus):
        """
        Finds unigrams occurring more than N times

        :param corpus: EITHER a function that yields corpus chunks or a corpus of text.
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

        if self.hash_values:
            duplicated_unigrams = {hash(unigram) for unigram, count in unigram_counts.items()
                                   if count >= self.threshold}
        else:
            duplicated_unigrams = {unigram for unigram, count in unigram_counts.items() if count >= self.threshold}
        logger.info(f"Found {len(duplicated_unigrams)} unigrams occurring at least {self.threshold} time(s).")

        return duplicated_unigrams

    def _find_duplicated_ngrams(self, corpus, n, nminusonegrams):
        """
        Finds duplicated n_grams in a memory efficient way

        The process is not computationally, but is memory efficient. n_grams are only counted if both 'halves' of an
        ngram are  present in the nminusonegrams list, for example (New York Times) would only be counted if both
        (New York) and (York Times) were in the set nminusonegrams. This prevents the list of ngrams from becoming
        staggeringly large.

        :param corpus: EITHER a function generating lists of tokenized documents or a list of tokenized documents.
        :param nminusonegrams: A set of n-1 grams to consider. For the return value to be accurate these must appear at
                               least n times in the text.
        :return: A set of ngrams that appear at least n times in the corpus.
        """
        ngram_counts = Counter()

        if callable(corpus):
            for i, corpus_chunk in enumerate(corpus()):
                start_time = time.perf_counter()
                chunk_counts = find_ngram_counts(corpus=corpus_chunk,
                                                 n=n,
                                                 check_against=nminusonegrams,
                                                 use_hashing=self.hash_values,
                                                 join_char=self.join_char)
                logger.info(f"Chunk {i}: Found {len(chunk_counts)} {n}-grams in "
                            f"{time.perf_counter() - start_time:.2f} seconds.")
                ngram_counts.update(chunk_counts)
        else:
            ngram_counts = find_ngram_counts(corpus=corpus,
                                             n=n,
                                             check_against=nminusonegrams,
                                             use_hashing=self.hash_values,
                                             join_char=self.join_char)

        duplicated_ngrams = {ngram for ngram, count in ngram_counts.items() if count >= self.threshold}
        logger.info(f"Found {len(duplicated_ngrams)} {n}-grams occurring at least {self.threshold} time(s).")
        return duplicated_ngrams

    def clean_text(self, corpus: Iterable[List[str]], duplicated_ngrams: Set[str], threshold: float, mode: str) \
            -> List[str]:
        """
        Removes documents with a high ratio of duplicated text

        In first mode keep an running record of duplicated n_grams that have been encountered already.

        :param corpus: An iterable of tokenized documents
        :param mode: Either 'first' or 'all'. Behaviour when encountering duplicated text. If 'first' then will keep the
                     first occurrence encountered, else if set to all will remove all occurrences seen.
        :param duplicated_ngrams: Set of duplicated ngrams
        :param threshold: If resemblance with duplicated text is above this then remove this document.
        :return: An iterator of cleaned documents.
        """
        if mode == 'first':
            seen_n_grams = set()

            for document in corpus:
                doc_ngrams = set(get_n_grams(document, self.n_gram, self.hash_values, self.join_char))
                resemblance = calc_resemblance(doc_ngrams, seen_n_grams)
                seen_n_grams.update(doc_ngrams.intersection(duplicated_ngrams))

                if resemblance >= threshold:
                    yield ''
                else:
                    yield ' '.join(document)

        elif mode == 'all':
            for document in corpus:
                doc_ngrams = set(get_n_grams(document, self.n_gram, self.hash_values, self.join_char))
                resemblance = calc_resemblance(doc_ngrams, duplicated_ngrams)

                if resemblance >= threshold:
                    yield ''
                else:
                    yield ' '.join(document)
        else:
            raise NotImplementedError(f"Mode {mode} not recognised. Must be either 'first' or 'all'")

    def clean_text_in_blocks(self, corpus: Iterable[str], blockizer, tokenizer, duplicated_ngrams: Set[str],
                             threshold: float, mode: str) -> Iterable[str]:
        """
        Removes low quality blocks from text

        Splits a document into blocks using the provided blockizer (or paragraph splitter), then finds the number of
        duplicated n_grams in that block. If the resemblance between the block and the duplicate set is above the
        threshold that block will be removed. If all text is of low quality will return an empty string.

        :type tokenizer:
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
                    doc_ngrams = set(get_n_grams(tokens, self.n_gram, self.hash_values, self.join_char))
                    resemblance = calc_resemblance(doc_ngrams, seen_n_grams)
                    seen_n_grams.update(doc_ngrams.intersection(duplicated_ngrams))

                    if resemblance >= threshold:
                        yield ''
                    else:
                        yield ' '.join(document)

        elif mode == 'all':
            for document in corpus:
                tokens = tokenizer(document)
                doc_ngrams = set(get_n_grams(tokens, self.n_gram, self.hash_values, self.join_char))
                resemblance = calc_resemblance(doc_ngrams, duplicated_ngrams)

                if resemblance >= threshold:
                    yield ''
                else:
                    yield ' '.join(document)
        else:
            raise NotImplementedError(f"Mode {mode} not recognised. Must be either 'first' or 'all'")


def find_unigram_counts(corpus):
    """Returns a count of unigrams in the corpus"""
    counter = Counter()
    for doc in corpus:
        counter.update(doc)
    return counter


def find_ngram_counts(corpus, n, check_against, use_hashing, join_char):
    """Returns a count of ngrams in the corpus"""
    counter = Counter()
    for doc in corpus:
        counter.update(get_n_grams_w_checking(doc, n, check_against, use_hashing, join_char))
    return counter


def get_n_grams_w_checking(tokens, n, check_against, use_hashing, join_char: str) -> List:
    assert n > 1, "This won't work for unigrams as no components"

    n_grams = []

    lhs = join_tokens(tokens[0:n - 1], use_hashing, join_char)

    for i in range(len(tokens) - n + 1):
        rhs = join_tokens(tokens[(i + 1):(i + n)], use_hashing, join_char)
        if (lhs in check_against) and (rhs in check_against):
            n_grams.append(join_tokens(tokens[i:(i + n)], use_hashing, join_char))
        lhs = rhs  # Saves calculating next time

    return n_grams


def join_tokens(tokens, hash_values, join_char):
    if hash_values:
        return hash(join_char.join(tokens))
    else:
        return join_char.join(tokens)


def get_n_grams(tokens: List[str], n: int, hash_values: bool, join_char: str) -> List[str]:
    return [join_tokens(tokens[i:i + n], hash_values, join_char) for i in range(len(tokens) - n + 1)]


def calc_resemblance(doc_ngrams: Set, comparison_ngrams: Set) -> float:
    """
    Discover ratio of document n_grams present in the target set
    """
    n_ngrams = len(doc_ngrams)
    n_in_both = len(doc_ngrams.intersection(comparison_ngrams))
    return n_in_both / n_ngrams


def simple_blockizer(text: str) -> List[str]:
    """Splits text at paragraph markers"""
    return NEWLINES_RE.split(text)


def simple_tokenizer(sentence: str) -> List[str]:
    """Splits text at whitespace markers"""
    return TOKENIZE_RE.split(sentence)

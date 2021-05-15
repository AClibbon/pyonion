import logging
import re
import time
from abc import ABC
from enum import Enum
from typing import Set, Iterable, List, Tuple

from .utils import (find_unigram_counts, find_ngram_counts, get_n_grams, calc_resemblance, simple_tokenizer,
                    simple_blockizer, scrub_ngrams)

logger = logging.getLogger(__name__)

BLOCK_JOIN_CHAR = '\n\n'


class CorpusProvider(ABC):
    """A class to provide data to the duplication remover"""

    def __init__(self, tokenizer=None, blockizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else simple_tokenizer
        self.blockizer = blockizer if blockizer is not None else simple_blockizer

    def iter_docs(self) -> Iterable[str]:
        """An iterator over documents"""
        pass

    def iter_tokens(self) -> Iterable[List[str]]:
        """An iterator over tokenized documents"""
        pass

    def iter_blocks(self) -> Iterable[List[str]]:
        """An iterator over document blocks (usually paragraphs)"""
        pass


class ListCorpusProvider(CorpusProvider):
    """A simple corpus provider to repeatedly iterate over a list"""

    def __init__(self, corpus: List[str], tokenizer=None, blockizer=None):
        super().__init__(tokenizer, blockizer)
        self.corpus = corpus

    def iter_docs(self) -> Iterable[str]:
        for doc in self.corpus:
            yield doc

    def iter_tokens(self) -> Iterable[List[str]]:
        for doc in self.corpus:
            yield self.tokenizer(doc)

    def iter_blocks(self) -> Iterable[List[str]]:
        for doc in self.corpus:
            blocks = self.blockizer(doc)
            yield blocks


class FileCorpusProvider(CorpusProvider):
    """Provide a corpus from a text file containing documents in the format:
    <d>Document text<\\d>
    <d>Next
    possibly multiline
    document<\\d>
    This is my own format."""
    DOCUMENT_START_MARKER = '<d>'
    DOCUMENT_END_REGEX = re.compile(r'<\\d>\n?$')

    def __init__(self, filepath: str, tokenizer=None, blockizer=None):
        super().__init__(tokenizer, blockizer)
        self.filepath = filepath

    def _read_docs(self):
        with open(self.filepath, 'r') as f:
            doc = ''
            for line in f.readlines():
                if line.startswith(self.DOCUMENT_START_MARKER):
                    if len(doc) > 0:
                        raise AssertionError("Multiple starts seen")
                    line = line.replace(self.DOCUMENT_START_MARKER, '')
                if re.search(self.DOCUMENT_END_REGEX, line):
                    line = re.sub(self.DOCUMENT_END_REGEX, '', line)
                    doc += line
                    yield doc
                    doc = ''
                else:
                    doc += line

    def iter_docs(self) -> Iterable[str]:
        yield from self._read_docs()

    def iter_tokens(self) -> Iterable[List[str]]:
        for doc in self._read_docs():
            yield self.tokenizer(doc)

    def iter_blocks(self) -> Iterable[List[str]]:
        for doc in self._read_docs():
            blocks = self.blockizer(doc)
            yield blocks


class CleaningMode(Enum):
    FIRST = 1  # Keep the first instance encountered
    ALL = 2  # Remove all duplicated instances


class DuplicateRemover:
    def __init__(self, hash_values=False, join_char='_', n_gram=10, duplication_threshold=2):
        """
        Uses a corpus provider to discover repeated segments of text in the corpus.

        :param hash_values: If true then work using hashed values - extra computation cost but lower memory requirements
        :param join_char: Character used to join n-grams  e.g. [New, York, Times] -> New_York_Times
        :param n_gram: Look for duplicated shingles/ngrams of this length. 10 is a reasonable value.
        :param duplication_threshold: n-grams occurring more than this many times count as duplicates.
        """
        self.hash_values = hash_values
        self.join_char = join_char
        self.n_gram = n_gram
        self.threshold = duplication_threshold

    def find_duplicated_ngrams(self, corpus: CorpusProvider) -> Set[str]:
        """
        Efficiently find duplicated n_grams in a corpus of documents.

        :param corpus: A provider of documents
        :return: A set of strings which represent the discovered ngrams, joined by the concatenation character.
        """
        logger.info("Finding Unigrams")
        duplicated_ngrams = self._find_duplicated_unigrams(corpus)

        for i in range(2, self.n_gram + 1):
            start_time = time.perf_counter()
            logger.info(f"Finding {i}-grams")
            duplicated_ngrams = self._find_duplicated_ngrams(corpus, i, duplicated_ngrams)
            logger.info(f"Found {len(duplicated_ngrams)} duplicated {i}-grams in {time.perf_counter() - start_time:.2f}"
                        f" seconds.")
        return duplicated_ngrams

    def _find_duplicated_unigrams(self, corpus: CorpusProvider):
        """
        Finds unigrams occurring more than N times

        :param corpus: A provider of a tokenized corpus
        :return: The set of duplicated unigrams
        """
        unigram_counts = find_unigram_counts(corpus.iter_tokens())

        if self.hash_values:
            duplicated_unigrams = {hash(unigram) for unigram, count in unigram_counts.items()
                                   if count >= self.threshold}
        else:
            duplicated_unigrams = {unigram for unigram, count in unigram_counts.items() if count >= self.threshold}
        logger.info(f"Found {len(duplicated_unigrams)} unigrams occurring at least {self.threshold} time(s).")

        return duplicated_unigrams

    def _find_duplicated_ngrams(self, corpus: CorpusProvider, n: int, nminusonegrams: Set):
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
        ngram_counts = find_ngram_counts(corpus=corpus.iter_tokens(),
                                         n=n,
                                         check_against=nminusonegrams,
                                         use_hashing=self.hash_values,
                                         join_char=self.join_char)

        duplicated_ngrams = {ngram for ngram, count in ngram_counts.items() if count >= self.threshold}
        logger.info(f"Found {len(duplicated_ngrams)} {n}-grams occurring at least {self.threshold} time(s).")
        return duplicated_ngrams

    def iter_clean_text(self, corpus: CorpusProvider, duplicated_ngrams: Set[str], threshold: float,
                        mode: CleaningMode) -> Iterable[Tuple[str, float]]:
        """
        Removes documents with a high ratio of duplicated text

        :param corpus: Corpus provider
        :param mode: Either 'first' or 'all'. Behaviour when encountering duplicated text. If 'first' then will keep the
                     first occurrence encountered, else if set to all will remove all occurrences seen.
        :param duplicated_ngrams: Set of duplicated ngrams - only consider these when looking at resemblence.
        :param threshold: If resemblance with duplicated text is above this then remove this document.
        :return: An iterator of cleaned documents.
        """
        if mode is CleaningMode.FIRST:
            yield from self._clean_text_first(corpus, duplicated_ngrams, threshold)

        elif mode is CleaningMode.ALL:
            yield from self._clean_text_all(corpus, duplicated_ngrams, threshold)

    def _clean_text_all(self, corpus: CorpusProvider, duplicated_ngrams, threshold):
        """Remove ALL documents with high resemblance"""
        for document in corpus.iter_tokens():
            doc_ngrams = set(get_n_grams(document, self.n_gram, self.hash_values, self.join_char))
            resemblance = calc_resemblance(doc_ngrams, duplicated_ngrams)

            if resemblance >= threshold:
                yield '', resemblance
            else:
                yield ' '.join(document), resemblance

    def _clean_text_first(self, corpus: CorpusProvider, duplicated_ngrams, threshold):
        """Keep only the first seen version of each document"""
        seen_n_grams = set()
        for document in corpus.iter_tokens():
            doc_ngrams = set(get_n_grams(document, self.n_gram, self.hash_values, self.join_char))
            resemblance = calc_resemblance(doc_ngrams, seen_n_grams)
            seen_n_grams.update(doc_ngrams.intersection(duplicated_ngrams))

            if resemblance >= threshold:
                yield '', resemblance
            else:
                yield ' '.join(document), resemblance

    def iter_clean_text_by_ngram(self, corpus: CorpusProvider,
                               duplicated_ngrams: Set[str], threshold: float,
                        mode: CleaningMode) -> Iterable[str]:
        """
        Removes duplicated ngrams from text, leaves the rest

        :param corpus: Corpus provider
        :param mode: Either 'first' or 'all'. Behaviour when encountering duplicated text. If 'first' then will keep the
                     first occurrence encountered, else if set to all will remove all occurrences seen.
        :param duplicated_ngrams: Set of duplicated ngrams - only consider these when looking at resemblence.
        :param threshold: If resemblance with duplicated text is above this then remove this document.
        :return: An iterator of cleaned documents.
        """
        if mode is CleaningMode.FIRST:
            yield from self._clean_text_by_ngram_first(corpus, duplicated_ngrams, threshold)

        elif mode is CleaningMode.ALL:
            yield from self._clean_text_by_ngram_all(corpus, duplicated_ngrams, threshold)

    def _clean_text_by_ngram_first(self, corpus: CorpusProvider,
                                   duplicated_ngrams, threshold):
        """Remove ngrams if you have seen them before, but leave them the
        first time around """
        if self.hash_values:
            logger.warning("This might not work with hash_values")

        seen_n_grams = set()
        for document in corpus.iter_tokens():
            doc_ngrams = set(get_n_grams(document, self.n_gram,
                                        self.hash_values,
                                     self.join_char))
            resemblance = calc_resemblance(doc_ngrams, seen_n_grams)

            # re-construct the document but kick out anything that is a
            # duplicated ngram:
            ngrams_2_remove = [doc_ngram for doc_ngram in doc_ngrams if
                               doc_ngram in seen_n_grams]
            seen_n_grams.update(doc_ngrams.intersection(duplicated_ngrams))

            trimmed_text = scrub_ngrams(document, ngrams_2_remove, self.join_char)

            yield trimmed_text, resemblance



    def _clean_text_by_ngram_all(self, corpus: CorpusProvider,
                                   duplicated_ngrams, threshold):
        """Remove ALL ngrams that are duplicated"""
        if self.hash_values:
            logger.warning("This might not work with hash_values")
        for document in corpus.iter_tokens():
            doc_ngrams = set(get_n_grams(document, self.n_gram,
                                        self.hash_values,
                            self.join_char))
            resemblance = calc_resemblance(doc_ngrams, duplicated_ngrams)

            # re-construct the document but kick out anything that is a
            # duplicated ngram:
            ngrams_2_remove = [doc_ngram for doc_ngram in doc_ngrams if
                              doc_ngram in duplicated_ngrams]

            trimmed_text = scrub_ngrams(document, ngrams_2_remove, self.join_char)
            yield trimmed_text, resemblance


    def iter_clean_text_in_blocks(self, corpus: CorpusProvider, duplicated_ngrams: Set[str], threshold: float,
                                  mode: CleaningMode) -> Iterable[str]:
        """
        Removes low quality blocks from text

        Splits a document into blocks using the provided blockizer (or paragraph splitter), then finds the number of
        duplicated n_grams in that block. If the resemblance between the block and the duplicate set is above the
        threshold that block will be removed. If all text is of low quality will return an empty string.

        :param corpus:
        :param mode: Either 'first' or 'all'. Behaviour when encountering duplicated text. If 'first' then will keep the
                     first occurrence encountered, else if set to all will remove all occurrences seen.
        :param duplicated_ngrams: Set of duplicated ngrams
        :param threshold: Remove blocks with resemblance above this
        :return: A Iterator over cleaned text. If all blocks of text are poor quality will return an empty string.
        """
        if mode is CleaningMode.FIRST:
            yield from self._clean_blocks_first(corpus, duplicated_ngrams, threshold)

        elif mode is CleaningMode.ALL:
            yield from self._clean_blocks_all(corpus, duplicated_ngrams, threshold)
        else:
            raise NotImplementedError(f"Mode {mode} not recognised. Must be either 'first' or 'all'")

    def _clean_blocks_all(self, corpus: CorpusProvider, duplicated_ngrams, threshold):
        for blocks in corpus.iter_blocks():
            clean_blocks = []
            for block in blocks:
                block_tokens = corpus.tokenizer(block)
                block_ngrams = set(get_n_grams(block_tokens, self.n_gram, self.hash_values, self.join_char))
                resemblance = calc_resemblance(block_ngrams, duplicated_ngrams)

                if resemblance < threshold:
                    clean_blocks.append(block)
            yield BLOCK_JOIN_CHAR.join(clean_blocks)

    def _clean_blocks_first(self, corpus: CorpusProvider, duplicated_ngrams, threshold):
        seen_n_grams = set()
        for blocks in corpus.iter_blocks():
            clean_blocks = []
            for block in blocks:
                block_tokens = corpus.tokenizer(block)
                block_ngrams = set(get_n_grams(block_tokens, self.n_gram, self.hash_values, self.join_char))
                resemblance = calc_resemblance(block_ngrams, seen_n_grams)
                seen_n_grams.update(block_ngrams.intersection(duplicated_ngrams))

                if resemblance < threshold:
                    clean_blocks.append(block)
            yield BLOCK_JOIN_CHAR.join(clean_blocks)


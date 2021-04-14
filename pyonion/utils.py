import re
from collections import Counter
from typing import List, Set


NEWLINES_RE = re.compile(r"(?:[\n\r] *){2,}")
TOKENIZE_RE = re.compile(r"\W+")


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
    """Discover n-grams whose n-1-grams exist in the set check_against"""
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
    if n_ngrams == 0:
        return 0.
    else:
        n_in_both = len(doc_ngrams.intersection(comparison_ngrams))
        return n_in_both / n_ngrams


def simple_blockizer(text: str) -> List[str]:
    """Splits text at paragraph markers"""
    return [block.strip() for block in NEWLINES_RE.split(text) if len(block) > 0]


def simple_tokenizer(sentence: str) -> List[str]:
    """Splits text at whitespace markers"""
    return [token for token in TOKENIZE_RE.split(sentence) if len(token) > 0]

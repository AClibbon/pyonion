import pytest

from pyonion.utils import calc_resemblance, get_n_grams_w_checking, simple_tokenizer, simple_blockizer, join_tokens, \
    get_n_grams, find_unigram_counts, find_ngram_counts

resemblance_data = [
    ({'a', 'b', 'c', 'd'}, {'d', 'e', 'f'}, .25),
    ({'a', 'b', 'c', 'd'}, {'e', 'f'}, 0.),
    ({'a', 'b', 'c', 'd'}, {'a', 'b', 'c', 'd', 'e', 'f'}, 1.),
    (set(), {'a', 'b', 'c'}, 0.)
]
n_grams_checking_data = [
    (['the', 'cat', 'sat', 'on', 'the', 'mat'], 4, ['the_cat_sat', 'cat_sat_on', 'on_the_mat'], ['the_cat_sat_on']),
    (['the', 'cat', 'sat', 'on', 'the', 'mat'], 5, ['the_cat_sat_on'], []),
    (['the', 'cat', 'sat', 'on', 'the', 'mat'], 5, ['the_cat_sat_on', 'cat_sat_on_the', 'sat_on_the_mat'],
     ['the_cat_sat_on_the', 'cat_sat_on_the_mat'])
]


@pytest.mark.parametrize("set_a, set_b, r", resemblance_data)
def test_calc_resemblance(set_a, set_b, r):
    assert calc_resemblance(set_a, set_b) == r


@pytest.mark.parametrize("tokens, n, check_against, expected_ngrams", n_grams_checking_data)
def test_get_n_grams_w_checking(tokens, n, check_against, expected_ngrams):
    use_hashing = False
    join_char = '_'
    n_grams = get_n_grams_w_checking(tokens, n, check_against, use_hashing, join_char)
    assert n_grams == expected_ngrams


tokeniser_test_data = [
    ("I'm a test document", ["I", "m", "a", "test", "document"]),
    ("zxzxc", ["zxzxc"]),
    (".£$%", []),
    ("ワタシワアリクツデウ", ["ワタシワアリクツデウ"]),
    ("", [])
]


@pytest.mark.parametrize("doc, expected_tokens", tokeniser_test_data)
def test_simple_tokenizer(doc, expected_tokens):
    assert simple_tokenizer(doc) == expected_tokens


blockizer_test_data = [
    ("""I'm text.
    
    But in blocks!""", ["I'm text.", "But in blocks!"]),
    ("""""", []),
    ("""I use carriage returns\r\n\nand new lines\n \n together""",
     ["I use carriage returns", "and new lines", "together"]),
    ("""This is some text written in blocks
        
        This is a repeated block
        
        This is a clean block""",
     ["This is some text written in blocks", "This is a repeated block", "This is a clean block"]),
    ("This is some text written in blocks\nsome of my blocks only include single returns\n\nThis is a clean block",
     ["This is some text written in blocks\nsome of my blocks only include single returns", "This is a clean block"])
]


@pytest.mark.parametrize("doc, expected_blocks", blockizer_test_data)
def test_simple_blockizer(doc, expected_blocks):
    blocks = simple_blockizer(doc)
    assert blocks == expected_blocks


join_tokens_test_data = [
    (['some', 'tokens', 'here'], '_', False, 'some_tokens_here'),
    (['some', 'tokens', 'here'], '&', False, 'some&tokens&here'),
    (['some', 'tokens', 'here'], '_', True, hash('some_tokens_here')),
    ([], '_', False, ''),
    ([], '_', True, hash(''))
]


@pytest.mark.parametrize("tokens, join_char, hashing, expected_result", join_tokens_test_data)
def test_join_tokens(tokens, join_char, hashing, expected_result):
    assert join_tokens(tokens, hashing, join_char) == expected_result


get_ngrams_test_data = [
    (['the', 'cat', 'sat', 'on', 'the', 'rug'], 3, '_', False,
     ['the_cat_sat', 'cat_sat_on', 'sat_on_the', 'on_the_rug']),
    (['the', 'cat', 'sat', 'on', 'the', 'rug'], 3, '_', True,
     [hash(ngram) for ngram in ['the_cat_sat', 'cat_sat_on', 'sat_on_the', 'on_the_rug']])
]


@pytest.mark.parametrize("tokens, n_gram, join_char, hashing, expected_result", get_ngrams_test_data)
def test_get_n_grams(tokens, n_gram, join_char, hashing, expected_result):
    assert get_n_grams(tokens, n_gram, hashing, join_char) == expected_result


def test_find_unigram_counts():
    test_corpus = [
        ['cat', 'dog', 'i', '', '@', 'cat'],
        ['cat', 'dog', 'cat', 'cat'],
    ]
    counts = find_unigram_counts(test_corpus)

    assert counts['dog'] == 2
    assert counts['cat'] == 5


def test_find_ngram_counts():
    test_corpus = [
        ['cat', 'dog', 'cat', 'i', '', '@', 'cat'],
        ['cat', 'dog', 'cat', 'i', 'cat'],
    ]
    duplicated = {'cat', 'dog'}

    counts = find_ngram_counts(test_corpus, 2, duplicated, False, '_')
    assert counts['cat_dog'] == 2
    assert counts['dog_cat'] == 2
    assert 'cat_i' not in counts


def test_find_ngram_counts_trigram():
    test_corpus = [
        ['cat', 'dog', 'cat', 'i', '', '@', 'cat'],
        ['cat', 'dog', 'cat', 'i', 'cat'],
        ['cat', 'dog', 'mouse', 'i', 'cat'],
    ]
    duplicated = {'cat_dog', 'dog_cat'}

    counts = find_ngram_counts(test_corpus, n=3, check_against=duplicated, use_hashing=False, join_char='_')
    assert counts['cat_dog_cat'] == 2
    assert 'dog_cat_i' not in counts


def test_find_ngram_counts_hashed():
    test_corpus = [
        ['cat', 'dog', 'cat', 'i', '', '@', 'cat'],
        ['cat', 'dog', 'cat', 'i', 'cat'],
    ]
    duplicated = set([hash(token) for token in ['cat', 'dog']])

    counts = find_ngram_counts(test_corpus, 2, duplicated, use_hashing=True, join_char='_')
    assert counts[hash('cat_dog')] == 2
    assert counts[hash('dog_cat')] == 2
    assert hash('cat_i') not in counts

from pyonion.remover import DuplicateRemover, CleaningMode
from pyonion.utils import simple_blockizer, simple_tokenizer


def test_iter_clean_text_in_blocks_all():
    dr = DuplicateRemover(hash_values=False, join_char='_', n_gram=3, duplication_threshold=2)

    corpus = [
        """This is some text written in blocks
        
        This is a repeated block
        
        This is a clean block""",
        """A second piece of text
        
         This is a repeated block"""
    ]
    expected_cleaned = [
        """This is some text written in blocks\n\nThis is a clean block""",
        """A second piece of text"""
    ]
    corpus_tokens = [simple_tokenizer(doc) for doc in corpus]
    duplicated = dr.find_duplicated_ngrams(corpus_generator=corpus_tokens)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus, blockizer=simple_blockizer,
                                                                tokenizer=simple_tokenizer,
                                                                duplicated_ngrams=duplicated,
                                                                threshold=.5, mode=CleaningMode.ALL)]
    assert clean_corpus == expected_cleaned


def test_iter_clean_text_in_blocks_all_single_return():
    dr = DuplicateRemover(hash_values=False, join_char='_', n_gram=3, duplication_threshold=2)

    corpus = [
        """This is some text written in blocks\nsome of my blocks only include single returns
        
        This is a repeated block
        
        This is a clean block""",
        """A second piece of text
        
         This is a repeated block"""
    ]
    expected_cleaned = [
        "This is some text written in blocks\nsome of my blocks only include single returns\n\nThis is a clean block",
        """A second piece of text"""
    ]
    corpus_tokens = [simple_tokenizer(doc) for doc in corpus]
    duplicated = dr.find_duplicated_ngrams(corpus_generator=corpus_tokens)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus, blockizer=simple_blockizer,
                                                                tokenizer=simple_tokenizer,
                                                                duplicated_ngrams=duplicated,
                                                                threshold=.5, mode=CleaningMode.ALL)]
    assert clean_corpus == expected_cleaned


def test_iter_clean_text_in_blocks_first():
    dr = DuplicateRemover(hash_values=False, join_char='_', n_gram=3, duplication_threshold=2)

    corpus = [
        """This is some text written in blocks
        
        This is a repeated block
        
        This is a clean block""",
        """A second piece of text
        
         This is a repeated block"""
    ]
    expected_cleaned = [
        """This is some text written in blocks\n\nThis is a repeated block\n\nThis is a clean block""",
        """A second piece of text"""
    ]
    corpus_tokens = [simple_tokenizer(doc) for doc in corpus]
    duplicated = dr.find_duplicated_ngrams(corpus_generator=corpus_tokens)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus, blockizer=simple_blockizer,
                                                                tokenizer=simple_tokenizer,
                                                                duplicated_ngrams=duplicated,
                                                                threshold=.5, mode=CleaningMode.FIRST)]
    assert clean_corpus == expected_cleaned


def test_iter_clean_text_in_blocks_first_hashed():
    dr = DuplicateRemover(hash_values=True, join_char='_', n_gram=3, duplication_threshold=2)

    corpus = [
        """This is some text written in blocks
        
        This is a repeated block
        
        This is a clean block""",
        """A second piece of text
        
         This is a repeated block"""
    ]
    expected_cleaned = [
        """This is some text written in blocks\n\nThis is a repeated block\n\nThis is a clean block""",
        """A second piece of text"""
    ]
    corpus_tokens = [simple_tokenizer(doc) for doc in corpus]
    duplicated = dr.find_duplicated_ngrams(corpus_generator=corpus_tokens)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus, blockizer=simple_blockizer,
                                                                tokenizer=simple_tokenizer,
                                                                duplicated_ngrams=duplicated,
                                                                threshold=.5, mode=CleaningMode.FIRST)]
    assert clean_corpus == expected_cleaned


def test_iter_clean_text_in_blocks_first_generator():
    dr = DuplicateRemover(hash_values=False, join_char='_', n_gram=3, duplication_threshold=2)

    def my_generator():
        for doc in [
            ["This is some text written in blocks\n\nThis is a repeated block\n\nThis is a clean block"],
            ["A second piece of text\n\nThis is a repeated block"]
        ]:
            yield doc

    expected_cleaned = [
        """This is some text written in blocks\n\nThis is a repeated block\n\nThis is a clean block""",
        """A second piece of text"""
    ]

    def my_token_generator():
        for doc in my_generator():
            yield simple_tokenizer(doc)

    duplicated = dr.find_duplicated_ngrams(corpus_generator=my_token_generator)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=my_generator(), blockizer=simple_blockizer,
                                                                tokenizer=simple_tokenizer,
                                                                duplicated_ngrams=duplicated,
                                                                threshold=.5, mode=CleaningMode.FIRST)]
    for clean, expected in zip(clean_corpus, expected_cleaned):
        assert clean == expected

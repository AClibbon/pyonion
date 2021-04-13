from pyonion.remover import DuplicateRemover, CleaningMode, ListCorpusProvider
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
    corpus_provider = ListCorpusProvider(corpus)

    duplicated = dr.find_duplicated_ngrams(corpus_provider)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus_provider,
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
    corpus_provider = ListCorpusProvider(corpus)

    duplicated = dr.find_duplicated_ngrams(corpus_provider)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus_provider,
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
    corpus_provider = ListCorpusProvider(corpus)

    duplicated = dr.find_duplicated_ngrams(corpus_provider)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus_provider,
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
    corpus_provider = ListCorpusProvider(corpus)

    duplicated = dr.find_duplicated_ngrams(corpus_provider)

    clean_corpus = [doc for doc in dr.iter_clean_text_in_blocks(corpus=corpus_provider,
                                                                duplicated_ngrams=duplicated,
                                                                threshold=.5, mode=CleaningMode.FIRST)]
    assert clean_corpus == expected_cleaned

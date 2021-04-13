"""
A rough Python implementation of http://corpus.tools/wiki/Onion.

Will be significantly slower, as does not use C++, or the Judy memory arrays, though is suitable for smaller corpora.
On a machine with 64Gb RAM available a corpus of 20 million records containing 2 billion tokens was processed in roughly
24 hours! This includes having to read the corpus from memory multiple times.
"""
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Don't log by default



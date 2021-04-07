"""
A rough Python implementation of http://corpus.tools/wiki/Onion.

Will be significantly slower, as does not use C++, or the Judy memory arrays, though is suitable for smaller corpora.
"""
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Don't log by default



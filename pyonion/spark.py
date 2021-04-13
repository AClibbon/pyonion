def get_duplicated_unigrams(rdd, tokenizer, threshold):
    """Spark implementation of duplicated unigram detection"""
    return rdd.flatMap(lambda text: [t for t in tokenizer(text)]) \
        .map(lambda token: (token, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] >= threshold)




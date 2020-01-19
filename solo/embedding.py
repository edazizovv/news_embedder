
# ======================================================================================================================
"""
Functional interface is the following:
input: 'x' as text and '*args'
output: numpy.ndarray with shape (n, 1)

"""
# ======================================================================================================================

# flair

def flair_word_embeddings(codes):
    from flair.embeddings import WordEmbeddings, FlairEmbeddings

    layers = []
    for code in codes:
        embedding = None
        if code == 'glove':
            # with the ex.: torch.Tensor with shape (100, 10)
            embedding = WordEmbeddings('glove')
        if code == 'turian':
            # with the ex.: torch.Tensor with shape (50, 10)
            embedding = WordEmbeddings('turian')
        if code == 'extvec':
            # with the ex.: torch.Tensor with shape (300, 10)
            embedding = WordEmbeddings('extvec')
        if code == 'crawl':
            # with the ex.: torch.Tensor with shape (300, 10)
            embedding = WordEmbeddings('crawl')
        if code == 'news':
            embedding = WordEmbeddings('news')
        if code == 'twitter':
            embedding = WordEmbeddings('twitter')
        if code == 'en-wiki':
            embedding = WordEmbeddings('en-wiki')
        if code == 'en-crawl':
            embedding = WordEmbeddings('en-crawl')
        if code == 'en-forward':
            embedding = FlairEmbeddings('en-forward')
        if code == 'en-backward':
            embedding = FlairEmbeddings('en-backward')
        if code == 'news-forward':
            embedding = FlairEmbeddings('news-forward')
        if code == 'news-backward':
            embedding = FlairEmbeddings('news-backward')
        if code == 'mix-forward':
            embedding = FlairEmbeddings('mix-forward')
        if code == 'mix-backward':
            embedding = FlairEmbeddings('mix-backward')
        if embedding is None:
            raise KeyError("Insufficient vespine gas")
        layers.append(embedding)
    return layers

def flair_embeddings(x, *args):
    from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings, Sentence

    word_embedders, aggregating_strategy, aggregating_params = args[0], args[1], args[2]
    embedding = None
    if aggregating_strategy == 'pooling':
        # TODO: check if kwargs work
        embedding = DocumentPoolEmbeddings(word_embedders,
                                           **aggregating_params)
    if aggregating_strategy == 'rnn':
        # TODO: check if kwargs work
        embedding = DocumentRNNEmbeddings(word_embedders,
                                          **aggregating_params)
    if embedding is None:
        raise KeyError("Insufficient vespine gas")
    sentence = Sentence(x)
    embedding.embed(sentence)
    return sentence.embedding.detach().numpy().reshape(-1, 1)


# sister

def sister_embeddings(x, *args):
    import sister

    aggregating_strategy = args[0]
    embedding = None
    if aggregating_strategy == 'mean':
        embedding = sister.MeanEmbedding(lang="en")
    if embedding is None:
        raise KeyError("Insufficient vespine gas")
    return embedding(x)


# spacy


def spacy_embeddings(x, *args):
    import spacy

    model = args[0]
    embedder = None
    if model == 'sm':
        embedder = spacy.load('en_core_web_sm')
    if model == 'md':
        embedder = spacy.load('en_core_web_md')
    if model == 'lg':
        embedder = spacy.load('en_core_web_lg')
    if embedder is None:
        raise KeyError("Insufficient vespine gas")
    doc = embedder(x)
    return doc.vector


# Universal Sentence Encoder (USE)

def use_embeddings(x, *args):
    import tensorflow
    import tensorflow_hub

    x = [x]
    embed = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    result = embed(x)

    return result.numpy().reshape(-1, 1)


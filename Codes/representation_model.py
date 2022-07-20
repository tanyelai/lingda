from simpletransformers.language_representation import RepresentationModel
from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords


def calc_bert_represenetations(texts, bert_checkpoint: str = 'dbmdz/bert-base-turkish-uncased', use_cuda: bool = True):
    """
    Calculates BERT representations using the model weights from `bert_checkpoint`

    Args:
        texts (array-like): texts to encode using BERT
        bert_checkpoint (str): HuggingFace-style checkpoint to use for encoding sentences
        (default is `dbmdz/bert-base-turkish-uncased`)
        use_cuda (bool): whether to use CUDA or not
        (default is False)

    returns:
        an array of the text embeddings
    """
    model = RepresentationModel(
        model_type='bert', model_name=bert_checkpoint, use_cuda=use_cuda)
    embeddings = model.encode_sentences(texts, combine_strategy='mean')
    return embeddings


def calc_w2v_representations(texts, keyed_vectors_path: str = None):
    """
    Calculates Word2vec representations of `texts`

    Args:
        texts (array-like): texts to encode using Word2vec
        keyed_vectors_path (str): file path to load KeyedVectors from. Trains Word2vec on `texts` if None
        (default is None)

    returns:
        an array of the text embeddings enocded using Word2vec
    """
    return None
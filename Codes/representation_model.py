import re
import numpy as np

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

def vectorize_sentence(sentence:str, word_vectors, stpwrds= stopwords.words('turkish')):
    sentence = re.sub(r"[^\w\s]", " ", sentence.lower()).split()
    words = np.array([word_vectors[word] for word in sentence if word in word_vectors and word not in stpwrds])
    if words.size:
        return np.mean(words, 0)
    else:
        return np.zeros(300,)

def calc_w2v_representations(texts, keyed_vectors_path: str = None, w2v_size: int = 300, w2v_window: int = 5,
                             w2v_min_count: int = 4, w2v_epoch: int = 16):
    """
    Calculates Word2vec representations of `texts`

    Args:
        texts (array-like): texts to encode using Word2vec
        keyed_vectors_path (str): file path to load KeyedVectors from. Trains Word2vec on `texts` if None
        (default is None)
        w2v_size (int): size of w2v vectors
        (default is 300)
        w2v_window (int): window size of w2v vectors
        (default is 5)
        w2v_epoch (int): number of epochs to train Word2vec
        (default is 16)
        w2v_min_count (int): min number of word occurrences to include a word
        (default is 4)

    returns:
        an array of the text embeddings enocded using Word2vec
    """
    if keyed_vectors_path is not None:
        word_vectors = KeyedVectors.load(keyed_vectors_path)
    else:
        documents = [text.split() for text in texts]
        w2v_model = Word2Vec(vector_size=w2v_size,
                             window=w2v_window,
                             min_count=w2v_min_count,
                             workers=8)
        w2v_model.build_vocab(documents)
        w2v_model.train(documents, total_examples=len(
            documents), epochs=w2v_epoch)

        word_vectors = w2v_model.wv

    vectorize = lambda t: vectorize_sentence(t, word_vectors)

    return np.array(list(map(vectorize)))

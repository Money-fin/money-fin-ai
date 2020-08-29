import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from preprocess import f_list

vocab = None
model = None


def get_vocab():
    global vocab
    if vocab is None:
        model, vocab = get_pytorch_kobert_model()
    return vocab


def get_model():
    global model
    if model is None:
        model, vocab = get_pytorch_kobert_model()

    return model


class PreProcessor(object):
    def __call__(self, text):
        for fn in f_list:
            text = fn(text)

        return text


class Embedding(object):
    MAX_LEN = 40
    
    @classmethod
    def _make_input_token(cls, text, *, sector=None):  # type: str, Union[str, None]
        p = PreProcessor()
        text = p(text)

        tok_path = get_tokenizer()
        sp = SentencepieceTokenizer(tok_path)
        tokenized_text = sp(text)

        tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
        token_type_ids = [0 for _ in range(len(tokenized_text))]
        if sector:
            tokenized_sector = sp(sector)
            tokenized_text += tokenized_sector + ['[SEP]']
            token_type_ids += [1 for _ in range(len(tokenized_sector) + 1)]
        
        # print(f'tokenized_text={tokenized_text}')
        # print(f'len={len(tokenized_text)}')  # TODO handle various lenghths
        
        assert len(tokenized_text) == len(token_type_ids)
        token_type_ids = torch.LongTensor([token_type_ids])

        return tokenized_text, token_type_ids  # type: List[str]

    @classmethod
    def _make_input_ids(cls, tokens):  # type: List[str]
        _vocab = get_vocab()
        tokens = [vocab[t] for t in tokens]
        ret = torch.LongTensor([tokens])  # TODO batchsize; 1
        return ret

    @classmethod
    def get_classification_vector(cls, text, sector):
        tokens, token_type_ids = cls._make_input_token(text, sector=sector)
        input_ids = cls._make_input_ids(tokens)

        _model = get_model()
        token_vectors, _ = _model(input_ids, token_type_ids=token_type_ids)

        return token_vectors[0][0]

    @classmethod
    def get_sentiment_vector(cls, text):
        tokens, _ = cls._make_input_token(text)
        input_ids = cls._make_input_ids(tokens)

        _model = get_model()
        token_vectors, _ = _model(input_ids)
        return token_vectors[0][0]

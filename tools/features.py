#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys
import torchlanguage.transforms
import os
import torch
from . import settings


# Create tokenizer
def create_tokenizer(tokenizer_type, lang="en_core_web_lg"):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
    # Tokenizer
    if tokenizer_type == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    elif tokenizer_type == "nltk-twitter":
        tokenizer = nsNLP.tokenization.NLTKTweetTokenizer()
    elif tokenizer_type == "spacy":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang)
    elif tokenizer_type == "spacy_wv":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang, original=True)
    else:
        sys.stderr.write(u"Unknown tokenizer type!\n")
        exit()
    # end if

    # Return tokenizer object
    return tokenizer
# end create_tokenizer


# Create transformer
def create_transformer(feature, embedding="", path="", lang="en_vectors_web_lg"):
    """
    Create the transformer
    :param feature:
    :param embedding:
    :param path:
    :param lang:
    :param n_gram:
    :return:
    """
    # ## Part-Of-Speech
    if "pos" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.PartOfSpeech(model=lang),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToOneHot(voc_size=16)
        ])
    # ## Function Words Embedding
    elif "fwv" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.FunctionWord(model=lang, join=True),
            torchlanguage.transforms.GloveVector(model=lang),
            torchlanguage.transforms.Reshape((-1, 300))
        ])
    # ## Function words
    elif "fw" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.FunctionWord(model=lang),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToOneHot(voc_size=300)
        ])
    # ## Tag
    elif "tag" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Tag(model=lang),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToOneHot(voc_size=45)
        ])
    # ## Word Vector bigram
    elif "wv2" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.GloveVector(model=lang),
            torchlanguage.transforms.ToNGram(n=2, overlapse=False),
            torchlanguage.transforms.Reshape((-1, 600))
        ])
    # ## Word Vector
    elif "wv" in feature:
        if embedding == 'glove':
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.GloveVector(model=lang),
                torchlanguage.transforms.Reshape((-1, 300))
            ])
        elif embedding == 'word2vec':
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.GensimModel(
                    model_path=os.path.join(path, embedding, "embedding.en.bin")
                )
            ])
        elif embedding == 'fasttext':
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.GensimModel(
                    model_path=os.path.join(path, embedding, "embedding.en.vec")
                )
            ])
        # end if
    # Sense2Vec
    elif "s2v" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Sense2Vec(path=settings.s2v_path),
            torchlanguage.transforms.Reshape((-1, settings.s2v_embedding_dim))
        ])
    # ## Character embedding
    elif "c1" in feature:
        token_to_ix, embedding_weights = load_character_embedding(path)
        embedding_dim = embedding_weights.size(1)
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.Reshape((-1, embedding_dim))
        ])
    # ## Character 2-gram embedding
    elif "c2" in feature:
        token_to_ix, embedding_weights = load_character_embedding(path)
        embedding_dim = embedding_weights.size(1)
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character2Gram(overlapse=True),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.Reshape((-1, embedding_dim))
        ])
    # ## Character 3-gram embedding
    elif "c3" in feature:
        token_to_ix, embedding_weights = load_character_embedding(path)
        embedding_dim = embedding_weights.size(1)
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character3Gram(overlapse=True),
            torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
            torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
            torchlanguage.transforms.Reshape((-1, embedding_dim))
        ])
    else:
        raise NotImplementedError(u"Feature type {} not implemented".format(feature))
    # end if
# end create_transformer


# Load character embedding
def load_character_embedding(emb_path):
    """
    Load character embedding
    :param emb_path:
    :return:
    """
    token_to_ix, weights = torch.load(open(emb_path, 'rb'))
    return token_to_ix, weights
# end load_character_embedding

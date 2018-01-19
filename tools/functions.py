#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys
import pickle
import os


# Create tokenizer
def create_tokenizer(tokenizer_type):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
    # Tokenizer
    if tokenizer_type == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    elif tokenizer_type == "spacy":
        tokenizer = nsNLP.tokenization.SpacyTokenizer()
    elif tokenizer_type == "spacy_wv":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(original=True)
    else:
        sys.stderr.write(u"Unknown tokenizer type!\n")
        exit()
    # end if

    # Return tokenizer object
    return tokenizer
# end create_tokenizer


# Create converter
def create_converter(converters_desc, pca_path, voc_size, uppercase, alphabet, fill_in=True):
    """
    Create converter
    :param converters_desc:
    :return:
    """
    # Converter list
    converter_list = list()

    # Joined converters
    joined_converters = True if len(converters_desc) > 1 else fill_in

    # For each converter
    for converter_desc in converters_desc:
        # Converter's info
        converter_type = converter_desc[0]
        converter_size = -1 if len(converter_desc) == 1 else converter_desc[1]

        # PCA model
        if converter_size != -1:
            pca_model = pickle.load(
                open(os.path.join(pca_path, converter_type + unicode(converter_size) + u".p"), 'r'))
        else:
            pca_model = None
        # end if

        # Choose a text to symbol converter.
        if converter_type == "pos":
            converter = nsNLP.esn_models.converters.PosConverter(pca_model=pca_model, fill_in=joined_converters)
        elif converter_type == "tag":
            converter = nsNLP.esn_models.converters.TagConverter(pca_model=pca_model, fill_in=joined_converters)
        elif converter_type == "fw":
            converter = nsNLP.esn_models.converters.FuncWordConverter(pca_model=pca_model, fill_in=joined_converters)
        elif converter_type == "wv":
            converter = nsNLP.esn_models.converters.WVConverter(pca_model=pca_model, fill_in=joined_converters)
        elif converter_type == "oh":
            converter = nsNLP.esn_models.converters.OneHotConverter(voc_size=voc_size, uppercase=uppercase)
        elif converter_type == "ch":
            converter = nsNLP.esn_models.converters.LetterConverter(alphabet=alphabet)
        else:
            raise Exception(u"Unknown converter type {}".format(converter_desc))
        # end if

        # Add to list
        converter_list.append(converter)
    # end for
# end create_converter

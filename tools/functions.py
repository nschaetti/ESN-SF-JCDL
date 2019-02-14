#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import sys
import pickle
import os
import echotorch.nn as etnn


# Manage W
def manage_w(xp, args, keep_w):
    """
    Manage W
    :param xp:
    :param args:
    :param keep_w:
    :return:
    """
    # First params
    rc_size = int(args.get_space()['reservoir_size'][0])
    rc_w_sparsity = args.get_space()['w_sparsity'][0]

    # Create W matrix
    w = etnn.ESNCell.generate_w(rc_size, rc_w_sparsity)

    # Save classifier
    if keep_w:
        xp.save_object(u"w", w)
    # end if

    return w
# end manage_w


# Get params
def get_params(space):
    """
    Get params
    :param space:
    :return:
    """
    # Params
    reservoir_size = int(space['reservoir_size'])
    w_sparsity = space['w_sparsity']
    leak_rate = space['leak_rate']
    input_scaling = space['input_scaling']
    input_sparsity = space['input_sparsity']
    spectral_radius = space['spectral_radius']
    ridge_param = space['ridge_param']
    feature = space['transformer'][0][0]
    aggregation = space['aggregation'][0][0]
    state_gram = space['state_gram']
    feedbacks_sparsity = space['feedbacks_sparsity']
    lang = space['lang'][0][0]
    embedding = space['embedding'][0][0]
    washout = int(space['washout'])

    return reservoir_size, w_sparsity, leak_rate, input_scaling, input_sparsity, spectral_radius, feature, aggregation, \
           state_gram, feedbacks_sparsity, lang, embedding, ridge_param, washout
# end get_params


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


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

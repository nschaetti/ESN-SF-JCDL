#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import torch
import settings


#########################################
# Dataset
#########################################


# Load dataset
def load_dataset(author, load_type, shuffle=False, remove_authors=False):
    """
    Load dataset
    :return:
    """
    # Remove
    if remove_authors:
        remove_text = [u"Isaac", u"isaac", u"ISAAC", u"Asimov", u"ASIMOV", u"asimov", u"Dick", u"DICK", u"dick",
                       u"Philip", u"PHILIP", u"philip", u"Robert", u"ROBERT", u"robert", u"Silverberg",
                       u"SILVERBERG", u"silververg"]
    else:
        remove_text = []
    # end if

    # Load from directory
    sfgram_dataset = torchlanguage.datasets.SFGramDataset(
        author=author,
        download=True,
        load_type=load_type,
        remove_texts=remove_text
    )

    # SFGram dataset training
    sfgram_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(sfgram_dataset),
        batch_size=1,
        shuffle=shuffle
    )

    # SFGram dataset test
    sfgram_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(sfgram_dataset, train=False),
        batch_size=1,
        shuffle=shuffle
    )
    return sfgram_dataset, sfgram_loader_train, sfgram_loader_test
# end load_dataset

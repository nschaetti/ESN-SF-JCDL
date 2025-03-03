#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import torch


#########################################
# Argument parsing
#########################################

# ESN training argument
def parser_esn_training():
    """
    ESN training argument
    :return:
    """
    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str, default="data/",
                      help="JSON file with the file description for each authors", required=False, extended=False)
    args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
    args.add_argument(command="--inverse-dev-test", name="inverse_dev_test", action='store_true', help="Inverse dev and test set?", extended=False, default=False)
    args.add_argument(command="--seed", name="seed", type=int, help="Random number generator initialisation", extended=False, default=1)

    # Author parameters
    args.add_argument(command="--author", name="author", type=str, help="Author to test", extended=False)
    args.add_argument(command="--novels", name="novels", action='store_true', help="Add novels ?", default=False, extended=False)
    args.add_argument(command="--remove_authors", name="remove_authors", action='store_true', help="Remove author in text", default=False, extended=False)

    # ESN arguments
    args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                      required=True, extended=True)
    args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                      default="1.0", extended=True)
    args.add_argument(command="--leak-rate", name="leak_rate", type=str, help="Reservoir's leak rate", extended=True,
                      default="1.0")
    args.add_argument(command="--input-scaling", name="input_scaling", type=str, help="Input scaling", extended=True,
                      default="0.5")
    args.add_argument(command="--input-sparsity", name="input_sparsity", type=str, help="Input sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--w-sparsity", name="w_sparsity", type=str, help="W sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--transformer", name="transformer", type=str,
                      help="The text transformer to use (fw, pos, tag, wv, c1, c2, c3, cnn)", default='wv',
                      extended=True)
    args.add_argument(command="--ridge-param", name="ridge_param", type=str, help="Ridge regression regularisation", extended=True,
                      default="0.0")
    args.add_argument(command="--pca-path", name="pca_path", type=str, help="PCA model to load", default=None,
                      extended=False)
    args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                      extended=False)
    args.add_argument(command="--aggregation", name="aggregation", type=str, help="Output aggregation method",
                      extended=True,
                      default="average")
    args.add_argument(command="--state-gram", name="state_gram", type=str, help="State-gram value",
                      extended=True, default="1")
    args.add_argument(command="--voc-size", name="voc_size", type=int, help="Voc. size",
                      default=30000, extended=False)
    args.add_argument(command="--feedbacks", name="feedbacks", action='store_true', help="Use feedbacks?",
                      default=False, extended=False)
    args.add_argument(command="--feedbacks-sparsity", name="feedbacks_sparsity", type=str, help="Feedbacks sparsity",
                      extended=True,
                      default="0.05")
    args.add_argument(command="--n-layers", name="n_layers", type=int, help="Number of layers in a stacked ESN", extended=True,
                      default="1")
    args.add_argument(command="--washout", name="washout", type=str, help="Washout period", extended=True, default="0")

    # Tokenizer and word vector parameters
    args.add_argument(command="--tokenizer", name="tokenizer", type=str,
                      help="Which tokenizer to use (spacy, nltk, spacy-tokens)", default='nltk', extended=False)
    args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters",
                      default='en_vectors_web_lg', extended=True)
    args.add_argument(command="--embedding", name="embedding", type=str,
                      help="Which word embedding to use? (glove, word2vec, skipgram, pretrained)",
                      default='glove', extended=True)
    args.add_argument(command="--embedding-path", name="embedding_path", type=str, help="Embedding directory",
                      default='~/Projets/TURING/Datasets/', extended=False)

    # Experiment output parameters
    args.add_argument(command="--window-size", name="window_size", type=str, help="Window size for prediction",
                      extended=True, required=False, default=0)
    args.add_argument(command="--measure", name="measure", type=str, help="Which measure to test (global/local)", extended=False, required=False, default='global')
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--sentence", name="sentence", action='store_true',
                      help="Test sentence classification rate?", default=False, extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)
    args.add_argument(command="--cuda", name="cuda", action='store_true',
                      help="Use CUDA?", default=False, extended=False)
    args.add_argument(command="--certainty", name="certainty", type=str, help="Save certainty data", default="", extended=False)

    # Parse arguments
    args.parse()

    # CUDA
    use_cuda = torch.cuda.is_available() if args.cuda else False

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager \
            (
            args.output,
            args.name,
            args.description,
            args.get_space(),
            args.n_samples,
            args.k,
            verbose=args.verbose
        )

    return args, use_cuda, param_space, xp
# end parser_esn_training

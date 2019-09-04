
# Import
import ktrain
from ktrain import text
import numpy as np
import os
import codecs
import argparse
import re


##################################################
# FUNCTIONS
##################################################


# Compute truth for a document
def compute_truth_for_document(doc_text, author):
    """
    Cmpute truth for a document
    :param doc_text:
    :param author:
    :return:
    """
    # Find all start
    start_matches = [(m.start(0), m.end(0)) for m in re.finditer(r'SFGRAM_START_{}'.format(author), doc_text)]
    stop_matches = [(m.start(0), m.end(0)) for m in re.finditer(r'SFGRAM_STOP_{}'.format(author), doc_text)]

    # Document truths
    document_truths = np.zeros(len(doc_text))

    # If the author is inside
    if len(start_matches) > 0:
        # For each part
        for j, (start1, start2) in enumerate(start_matches):
            # Ending
            (end1, end2) = stop_matches[j]

            # Put part to 1
            document_truths[start2:end1] = 1
        # end for
    # end if

    return document_truths
# end compute_truth_for_document


##################################################
# MAIN
##################################################

# Classes
classes = ['false', 'true']

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--author", type=str, default="ASIMOV")
parser.add_argument("--k", default=1)
parser.add_argument("--skip", type=int, default=1536)
args = parser.parse_args()

# Average accuracy
average_accuracy = np.zeros(args.k)

# For each fold
for k in range(args.k):
    # Author directory
    author_dir = os.path.join(args.datadir, args.author)

    # Fold directory
    fold_dir = os.path.join(author_dir, "k{}".format(k))

    # Train, test, val
    train_dir = os.path.join(fold_dir, "train")
    test_dir = os.path.join(fold_dir, "test")
    val_dir = os.path.join(fold_dir, "val")

    # Load training and validation data from a folder
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(
        fold_dir,
        maxlen=512,
        preprocess_mode='bert',
        classes=classes
    )

    # Load BERT
    learner = ktrain.get_learner(
        text.text_classifier('bert', (x_train, y_train)),
        train_data=(x_train, y_train),
        val_data=(x_test, y_test),
        batch_size=16
    )

    # Get good learning rate
    learner.lr_find()

    # Plot
    learner.lr_plot()

    # Train the model
    learner.fit(2e-5, 20, early_stopping=5)
    # learner.fit_onecycle(2e-5, 1)

    # Get the predictor
    predictor = ktrain.get_predictor(learner.model, preproc)

    # F-1 score per threshold
    fscore_per_threshold = np.zeros(20)

    # Thresholds
    thresholds = np.arange(0.0, 1.0, 20)

    # For each threshold between 0.0 and 1.0
    for i, threshold in enumerate(thresholds):
        # List of prediction and truths
        predicted_class = list()
        truth_class = list()

        # For false and true
        for test_class in ["false", "true"]:
            # Final dir
            final_dir = os.path.join(test_dir, test_class)

            # For each file in test
            for test_file in os.listdir(test_dir):
                if test_file[-4:] == ".txt":
                    # Read the file
                    document_text = codecs.open(os.path.join(final_dir, test_file), 'r', encoding='utf-8')

                    # Predict class
                    pred = predictor.predict([document_text])

                    # Above theshold ?
                    if pred[0, 1] > threshold:
                        predicted_class.append(1)
                    else:
                        predicted_class.append(0)
                    # end if

                    # Truth class
                    if test_class == "false":
                        truth_class.append(0)
                    else:
                        truth_class.append(1)
                    # end if
                # end if
            # end for
        # end for

        # Predicted class vector, truth class vector
        predicted_class_vector = np.array(predicted_class)
        truth_class_vector = np.array(truth_class)

        # Confusion matrix
        tp_fp = float(np.sum(predicted_class_vector))
        tp_fn = float(np.sum(truth_class_vector))
        tp = float(np.sum(truth_class_vector[predicted_class_vector]))

        # Precision and recall
        precision = tp / tp_fp
        recall = tp / tp_fn

        # Compute F1
        f1_test_scores = 2.0 * ((precision * recall) / (precision + recall))

        # Save
        fscore_per_threshold[i] = f1_test_scores
    # end for

    # Get the best threshold
    best_threshold = thresholds[np.argmax(fscore_per_threshold)]

    # Average F1-score
    average_f1_score = 0
    f1_score_count = 0

    # For each validation files
    for val_file in os.listdir(val_dir):
        if val_file[-4:] == ".txt":
            # Read the file
            document_text = codecs.open(os.path.join(val_dir, val_file), 'r', encoding='utf-8')

            # Document probabilities
            document_probs = np.zeros(len(document_text))

            # For each part
            for pos in range(0, len(document_text), args.skip):
                # Part text
                part_text = document_text[pos:pos+args.skip*2]

                # Predict class
                pred = predictor.predict([part_text])

                # Add probs
                document_probs[pos:pos+args.skip*2] += pred[0, 1]
            # end for

            # Normalized
            document_probs /= 2.0

            # Document classes
            document_predicted_classes = document_probs >= best_threshold
            document_predicted_classes[document_predicted_classes] = 1
            document_predicted_classes[not document_predicted_classes] = 0

            # Compute document truths
            document_truth_classes = compute_truth_for_document(document_text, args.author)

            # Confusion matrix
            tp_fp = float(np.sum(document_predicted_classes))
            tp_fn = float(np.sum(document_truth_classes))
            tp = float(np.sum(document_truth_classes[document_predicted_classes]))

            # Precision and recall
            precision = tp / tp_fp
            recall = tp / tp_fn

            # Compute F1
            f1_val_score = 2.0 * ((precision * recall) / (precision + recall))

            # Save
            average_f1_score += f1_val_score
            f1_score_count += 1
        # end if
    # end for

    # Show average F-1
    print("AVERAGE F-1 SCORE : {}".format(average_f1_score / f1_score_count))
# end for

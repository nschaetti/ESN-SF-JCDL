#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import argparse
import os
import codecs
import re


def save_part(base_path, findex, pindex, text):
    """
    Save part
    :param base_path:
    :param index:
    :param text:
    :return:
    """
    codecs.open(os.path.join(base_path, "{}-{}.txt".format(findex, pindex)), 'w', encoding='utf-8').write(text)
# end save_part


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--k", type=int)
parser.add_argument("--skip", type=int, default=256)
parser.add_argument("--train", type=int, default=50)
parser.add_argument("--test", type=int, default=20)
parser.add_argument("--val", type=int, default=21)
args = parser.parse_args()

# Create directory
for author in ["ASIMOV", "DICK", "SILVERBERG"]:
    # Author directory
    author_directory = os.path.join(args.output, author)
    if not os.path.exists(author_directory):
        os.mkdir(author_directory)
    # end if

    # For each fold
    for k in range(args.k):
        # Fold directory
        fold_directory = os.path.join(author_directory, "{}".format(k))

        # Create fold directory
        if not os.path.exists(fold_directory):
            os.mkdir(fold_directory)
        # end if

        # Train/test/val directory
        for cross in ["train", "test", "val"]:
            cross_directory = os.path.join(fold_directory, cross)

            # Create cross directory
            if not os.path.exists(cross_directory):
                os.mkdir(cross_directory)
            # end if

            # Classes
            true_directory = os.path.join(cross_directory, "true")
            false_directory = os.path.join(cross_directory, "false")

            # Create true class directory
            if not os.path.exists(true_directory):
                os.mkdir(true_directory)
            # end if

            # Create false class directory
            if not os.path.exists(false_directory):
                os.mkdir(false_directory)
            # end if
        # end for

        # Train/test/val directories
        train_directory = os.path.join(fold_directory, "train")
        test_directory = os.path.join(fold_directory, "test")
        val_directory = os.path.join(fold_directory, "val")

        # File index
        file_index = 0

        # For each file in dataset
        for data_file in os.listdir(args.dataset):
            # Text
            if data_file[-4:] == ".txt":
                # Log
                print(os.path.join(args.dataset, data_file))

                # Train and test
                if file_index < args.train + args.test:
                    # Truth directories
                    if file_index < args.train:
                        true_directory = os.path.join(train_directory, "true")
                        false_directory = os.path.join(train_directory, "false")
                    elif file_index >= args.train and file_index < args.train + args.test:
                        true_directory = os.path.join(test_directory, "true")
                        false_directory = os.path.join(test_directory, "false")
                    # end if

                    # Lis le fichier
                    document_text = codecs.open(os.path.join(args.dataset, data_file), 'r', encoding='utf-8').read()

                    # Find all start
                    start_matches = [(m.start(0), m.end(0)) for m in re.finditer(r'SFGRAM_START_{}'.format(author), document_text)]
                    stop_matches = [(m.start(0), m.end(0)) for m in re.finditer(r'SFGRAM_STOP_{}'.format(author), document_text)]

                    # If the author is inside
                    if len(start_matches) > 0:
                        # Part index
                        part_index = 0
                        class_part = "false"

                        # Save beginning
                        save_part(false_directory, file_index, part_index, document_text[:start_matches[0][0]])
                        part_index += 1

                        # For each part
                        for j, (start1, start2) in enumerate(start_matches):
                            # Log
                            print("Part {}".format(j))

                            # Change class
                            if class_part == "false":
                                class_part = "true"
                            else:
                                class_part = "false"
                            # end if

                            # Ending
                            (end1, end2) = stop_matches[j]

                            # Text
                            if class_part == "false":
                                final_directory = false_directory
                            else:
                                final_directory = true_directory
                            # end if

                            # For each segment
                            for pos in range(start2, end1, args.skip):
                                pos_end = pos + (args.skip * 2)
                                # Save beginning
                                if pos_end < end1:
                                    save_part(final_directory, file_index, part_index, document_text[pos:pos + (args.skip * 2)])
                                else:
                                    save_part(final_directory, file_index, part_index, document_text[pos:end1])
                                # end if
                                part_index += 1
                            # end for
                        # end for

                        # Save end
                        save_part(false_directory, file_index, part_index, document_text[:start_matches[0][0]])
                    else:
                        # Start, end
                        start, end = (0, len(document_text))

                        # Part index
                        part_index = 0

                        # For each segment
                        for pos in range(start, end, args.skip):
                            # Save beginning
                            save_part(false_directory, file_index, part_index, document_text[pos:pos+(args.skip*2)])
                            part_index += 1
                        # end for
                    # end if
                else:
                    # Lis le fichier
                    document_text = codecs.open(os.path.join(args.dataset, data_file), 'r', encoding='utf-8').read()

                    # Ecrit le fichier
                    codecs.open(os.path.join(val_directory, "{}.txt".format(file_index)), 'w', encoding='utf-8').write(document_text)
                # end if

                # Next file index
                file_index += 1
            # end if
        # end for
    # end for
# end for


#!/usr/bin/env python3

import argparse
import os
import pprint

from src.model_variants import vit_lite_7_4
from src.preprocessing import defaults
from src.pytorch_utils import ClassificationTester, RetrievalTester
import torch

MODEL_VARIANTS = {"vit-lite-7-4": (vit_lite_7_4, 256)}  # (function returning the model, dim)
CLASSIFICATION_SUPPORTED = ["cvl-1-1_with-enrollment_pages"]


def _argparse_weights(arg):
    _, ext = os.path.splitext(arg)
    if ext.lower() != ".pth":
        raise ValueError("File extension of saved model has to be .pth")
    return arg


def _argparse_classification(args):
    if args.trainset not in CLASSIFICATION_SUPPORTED or args.testset not in CLASSIFICATION_SUPPORTED or \
            args.trainset != args.testset:
        raise ValueError(
            f"When -c/--classification is provided, the positional arguments train-set and test-set have to be "
            f" both of the same value and the chosen dataset has to support classification-based evaluation. Supported "
            f"datasets: {','.join(CLASSIFICATION_SUPPORTED)}")


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="A command-line tool for evaluating Vision Transformers (ViTs) for writer identification and "
                    "writer retrieval.",
        epilog=f"The evaluation result is written to stdout.\n\nIn the standard setting of this tool, "
               f"writer recognition is considered as a retrieval task. For some "
               f"datasets, additionally an evaluation as a classification task can be performed (option "
               f"-c/--classification). Set the train-set as well as the test-set positional argument to one of "
               f"the following values: {','.join(CLASSIFICATION_SUPPORTED)} and provide option "
               f"-c/--classification if you wish to perform a classification-based evaluation in addition to the "
               f"retrieval-based evaluation.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-w", "--num-workers", type=int, default=8,
                        help="Number of PyTorch workers.")
    parser.add_argument("--model", choices=MODEL_VARIANTS.keys(), default="vit-lite-7-4",
                        help="Model variant.")
    parser.add_argument("--weights", required=True, type=_argparse_weights,
                        help="Path to the .pth file to be used. Has to be compatible with the training dataset "
                             "given as first positional argument.")
    parser.add_argument("--soft-top-k", nargs="+", type=int, choices=range(1, 11), default=[1, 2, 3, 4, 5],
                        help="Top k values to be evaluated with the ``soft criterion''. When option "
                             "-c/--classification is set, these values are used as well for the top k.")
    parser.add_argument("--hard-top-k", nargs="+", type=int, choices=range(1, 11), default=[1],
                        help="Top k values to be evaluated with the ``hard criterion''. Ignored, when --skip-retrieval "
                             "option is set")
    parser.add_argument("-c", "--classification", action='store_true',
                        help=f"Additionally perform classification-based evaluation. Only supported for the following "
                             f"datasets: {','.join(CLASSIFICATION_SUPPORTED)}")
    parser.add_argument("--skip-retrieval", action='store_true',
                        help=f"Do not perform retrieval-based evaluation. This option is only sensible, "
                             f"when the -c/--classification option is provided.")
    parser.add_argument("--metrics", nargs="+", choices=["canberra",
                                                         "chebyshev",
                                                         "cityblock",
                                                         "correlation",
                                                         "cosine",
                                                         "euclidean",
                                                         "seuclidean",
                                                         "sqeuclidean"], default=["cosine"],
                        help="Metrics to be used as distance measure for retrieval-based writer recognition. Ignored, "
                             "when --skip-retrieval option is set")
    # needed to get the number of classes the model was trained on
    parser.add_argument("trainset", choices=[k for k, v in defaults.items() if v.num_classes_train],
                        help="Dataset the used model (--saved-model) was trained on.")
    parser.add_argument("testset", choices=[k for k, v in defaults.items()],
                        help="Dataset to be evaluated.")

    args = parser.parse_args()
    if args.classification:
        _argparse_classification(args)

    return args


def main():
    args = _parse_arguments()

    # preprocess dataset to be evaluated
    test_set = defaults[args.testset]
    test_set()

    # prepare evaluation and evaluate
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes_train = defaults[args.trainset].num_classes_train
    model = MODEL_VARIANTS[args.model][0](num_classes_train).to(device=device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    print("Starting evaluation...")
    test_set_path = os.path.join("data", "preprocessed", args.testset, "test")

    if not args.skip_retrieval:
        print("Starting retrieval-based evaluation...")
        retrieval_tester = RetrievalTester((MODEL_VARIANTS[args.model][1], num_classes_train),
                                           test_set_path, model)
        res = retrieval_tester(device, 1, args.num_workers, sorted(list(set(args.soft_top_k))),
                               sorted(list(set(args.hard_top_k))), set(args.metrics))
        pprint.pprint(res)
    else:
        print("--skip-retrieval option provided - skipping retrieval-based evaluation")

    if args.classification:
        print("Starting classification-based evaluation...")
        classification_tester = ClassificationTester(test_set_path, model)
        res = classification_tester(device, 1, args.num_workers, sorted(list(set(args.soft_top_k))))
        pprint.pprint(res)


if __name__ == "__main__":
    main()

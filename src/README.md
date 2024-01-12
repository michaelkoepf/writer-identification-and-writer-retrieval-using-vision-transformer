# `src`
Contains various helper classes and functions.

## File Organisation
| File               | Description                                                                                                                                                                                                                                                    |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `datasets.py`      | Base and sub-classes representing writer identification and writer retrieval datasets. The base-class `WriterRecognitionDataset` can be extended to use new datasets.                                                                                          |
| `exceptions.py`    | Custom exceptions                                                                                                                                                                                                                                              |
| `lr_schedulers.py` | Custom learning rate schedulers                                                                                                                                                                                                            |
| `models.py`        | Helper methods to instantiate models                                                                                                                                                                                                                           |
| `preprocessing.py` | Classes for preprocessing writer recognition datasets. Base classes can be extended to implement different preprocessing methods. Furthermore, `defaults` contains predefined datasets incl. preprocessing steps used by the `train.py` and `eval.py` CLI tools |
| `pytorch_utils.py` | Utility methods for training and evaluation. These classes are used by the `train.py` and `eval.py` CLI tools                                                                                                                                                  |

For further information, have a look at the documentation in the source code of the respective files.
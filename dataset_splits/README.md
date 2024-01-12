# `dataset_splits`
This folder contains the used dataset splits. One dataset can have different splits (see also "Predefined Splits" below).

## File Format
The used format is `CSV`: `;` is used as separator and *no* header is used (i.e. the first line is already part of the payload).

The file format is as follows:
`<unique writer id>;<page/document id>;<path to the input file>;[train|val|test];<target directory>`

- `<path to the input file>` is the path relative to `data/raw/<dataset>`
- `<target directory>` is the path `data/preprocessed/<dataset split name>/[train|val|test]` relative to. If you use the provided `train.py` and `eval.py`, for `train` and `val` this value has to be set to the `<unique writer id>`, for `test` it has to be set to `<unique writer id>/<page/document id>`

Using the described format, you can create your own dataset splits. Currently supported and tested is only page-level input (i.e. one input file per writer id-page/document id combination).

For examples following this file format, you may have a look at the already predefined dataset splits included in this directory. For further information on the directory structure of `data` see `data/README.md`.


## Predefined Splits

We provide the following dataset splits for publicly available datasets.

| File | Description |   Max. *k* (hard criterion/retrieval-based) |   
| ------------- |-------------|-------------:|    
| `cvl-1-1_with-enrollment_pages.csv`      | Split of the `CVL` database for writer recognition *with enrollment* as proposed by He and Schomaker (page with ids 1, 2, 3 of each writer[\*] are used for training, the rest for testing). Additionally, we use the page with id 2 of each writer from their defined training set for validation. | 0 |   
| `cvl-1-1_with-enrollment_experiment_pages.csv`      | Subset of `cvl-1-1_with-enrollment_pages.csv` (first 50 writers). Used for model selection only. | 1 |   
| `cvl-1-1-test_retrieval-based-subset_with-enrollment_pages.csv`      | Contains only the test set of `cvl-1-1_with-enrollment_pages.csv` with exclusion of writer id 431. The exclusion is necessary for retrieval-based evaluation, since this writer has only one document in the test set. | 1 |   
| `cvl-1-1_without-enrollment_pages.csv`      | Original split of the `CVL` database. Page with id 2 of each writer in the training set is used for validation. | 2[\*\*] |   
| `icdar-2013_pages.csv`      | `ICDAR 2013` dataset. The pages used for validation are interleaved, with page 1, 3, 2, 4, 1, 3, 2, 4, ... being used for validation. This ensures a balanced validation set of english and greek texts, respectively.  | 3 |   
| `icdar-2013-test_greek-subset_pages.csv`      | Contains only the test set with pages written in greek script (pages with id 3 and 4) of `icdar-2013_pages.csv`. This subset is also referred to as "Greek language subset" | 1 |   
| `icdar-2013-test_latin-subset_pages.csv`      | Contains only the test set with pages written in latin script (pages with id 1 and 2) of `icdar-2013_pages.csv`. This subset is also referred to as "English language subset".  | 1 |   

[\*] writer with id 431 only wrote the text of page 1 and 2 and left page 3 blank   
[\*\*] due two writer with id 431, who left two pages blank


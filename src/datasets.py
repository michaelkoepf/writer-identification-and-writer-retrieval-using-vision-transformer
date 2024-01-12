import glob
import hashlib
import os
import urllib.request
import zipfile

import patoolib
from src.exceptions import InvalidHashError, IllegalArgumentError

"""
Classes representing writer recognition datasets with the following functionality:
- Download the datasets' archives from the internet (if publicly available)
- Verify the archives
- Extract the archives
- Preprocess the datasets using a customisable pipeline (see `preprocessing.py`)
"""


class WriterRecognitionDataset:
    """Base class for writer identification and writer retrieval datasets."""

    _DIRNAME_EXTRACTED_FILES = "raw"

    def __init__(self, identifier, filenames, filenames_url, filenames_sha256,
                 root_dir=os.path.join(os.curdir, "data"), download=True, transformation_pipeline=None, quiet=False):
        """
        Args:
            identifier: Unique, human-readable identifier of the dataset
            filenames (list): File names of the dataset's archives
            filenames_url (list): URLs, where the dataset's archives are publicly available on the internet for
            download.
            In case there are not available on the internet, provide a list with `None` entries.
            The length and the position of the list's entries needs to correspond with those in `filenames`.
            filenames_sha256 (list): List of SHA256 hashes of the archives in `filenames`. The
            length and the position of the list's entries needs to correspond with those in `filenames`.
            root_dir (optional): Root directory, where the archives, their extracted data and the pre-processed files
            should be stored (default: `./data`)
            download (optional): Indicates, if the archives given in `filenames_url` should be downloaded from the
            internet.
            If `True`, `filenames_url` has to contain entries with valid URLs. If a file with a filename given in
            `filenames`
            does already exist, it is not downloaded again.
            transformation_pipeline (optional): An instance of `TransformationPipeline` containing the preprocessing
            steps (see `preprocessing.py`)
            quiet (optional): Switches logging output during processing on/off
        """
        self.identifier = identifier
        self.filenames = filenames
        self.filenames_url = filenames_url
        self.filenames_sha256 = filenames_sha256
        self.root_dir = os.path.abspath(root_dir)
        self.download = download
        self.transformation_pipeline = transformation_pipeline
        self.quiet = quiet

    def _log(self, msg):
        if not self.quiet:
            print(msg)

    def download_archive(self):
        """
        Downloads the respective archives from this instance from the internet and saves them to `root_dir` with the
        names given in `filenames`.

        After successful download, the integrity of the archives is verified. If a file with the given filename already
        exists, the archive is not download again and is only verified.
        """
        self._log(f"Check if {self.identifier} dataset has already been downloaded previously...")

        for filename, sha256_hash, url in zip(self.filenames, self.filenames_sha256, self.filenames_url):
            file_path = os.path.join(self.root_dir, filename)

            if filename in map(os.path.basename, glob.glob(file_path)):  # file exists
                self._log(f"The dataset exists already ({file_path} exists).")
                self._verify_archive(file_path, sha256_hash)
            else:
                if not self.download:
                    self._log(f"[❌] Dataset has not been downloaded previously, but argument {self.download}.")
                    raise IllegalArgumentError(
                        "Dataset has not been downloaded previously. Please set argument 'download' to 'True' to "
                        "proceed or download the dataset manually.")
                else:
                    self._log(f"Downloading dataset from {url}...")

                    with urllib.request.urlopen(url) as data, open(file_path, 'wb') as out:
                        out.write(data.read())

                    self._log(f"[✅] Successfully downloaded file from {url} to {file_path}.")

                    self._verify_archive(file_path, sha256_hash)

    def _verify_archive(self, file_path, file_sha256):
        self._log(f"Verifying {file_path}")

        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                b = f.read(2048)
                if not b:
                    break
                h.update(b)

        if h.hexdigest() != file_sha256:
            self._log(f"[❌] Verification of {file_path} failed. Try do delete and re-download the file.")
            raise InvalidHashError(f"Hashes do not match. Expected: {h.hexdigest()}, got: {file_sha256}")
        else:
            self._log("[✅] Verification done. Hashes match.")

    def extract_archive(self):
        """
        Extracts the respective archives from this instance to `root_dir`/raw/`filename` (`filename` refers to the
        single entries in `filenames`).

        This method depends on the successful completion of `download_archive()`, thus the invocation of
        `download_archive()` has to precede the invocation of this method.
        """
        for filename in self.filenames:
            extract_path = os.path.join(self.root_dir, self._DIRNAME_EXTRACTED_FILES, filename.split(".")[0])

            self._log(f"Check if dataset {self.identifier} has already been extracted...")

            if extract_path in glob.glob(os.path.join(self.root_dir, self._DIRNAME_EXTRACTED_FILES, "*")):
                self._log(f"The dataset was already extracted ({extract_path} exists).")
                return

            self._log(f"Extracting {filename} to {extract_path}...")
            self._extract(os.path.join(self.root_dir, filename),
                          os.path.join(self.root_dir, self._DIRNAME_EXTRACTED_FILES))
            self._log(f"[✅] Extracted {filename} successfully.")

    def _extract(self, src_file, target):
        """
        Hook to be overwritten by subclasses. This method implements the extraction of the archives.

        For extracting zip and rar archives, helper functions are already provided in this file (see `_extract_zip(
        )`, `extract_rar()`).

        Args:
            src_file: Archive to be extracted
            outdir: Directory, where the content of `src_file` should be extracted to
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Downloads, extracts and preprocesses this instance.

        If any of these steps has already been executed previously, it will not be executed again.
        """
        self.download_archive()
        self.extract_archive()

        if self.transformation_pipeline:
            self._log(f"Preprocessing images...")
            if self.transformation_pipeline(self.root_dir,
                                            [os.path.join(self._DIRNAME_EXTRACTED_FILES, filename.split(".")[0]) for
                                             filename in self.filenames]):
                self._log(f"[✅] Preprocessed all images successfully.")
            else:
                self._log(
                    f"The dataset was already preprocessed ({self.transformation_pipeline.dirname_out} "
                    f"exists).")
        else:
            self._log(f"No TransformationPipeline provided. Skipping.")

        self._log(f"[✅] Done.")


class CVLCroppedDataset(WriterRecognitionDataset):
    """CVL database (cropped), version 1.1 [1] [2]

    References:
    [1] F. Kleber, S. Fiel, M. Diem, and R. Sablatnig, ‘CVL-DataBase: An Off-Line Database for Writer Retrieval,
    Writer Identification and Word Spotting’, in 2013 12th International Conference on Document Analysis and
    Recognition, Washington, DC, USA, Aug. 2013, pp. 560–564. doi: 10.1109/ICDAR.2013.117.
    [2] F. Kleber, S. Fiel, M. Diem, and R. Sablatnig, ‘CVL Database - An Off-line Database for Writer Retrieval,
    Writer Identification and Word Spotting’. Zenodo, Nov. 20, 2018. doi: 10.5281/ZENODO.1492267.
    """

    _META_DATA = {
        "identifier": "CVL cropped 1.1",
        "filenames": ["cvl-database-cropped-1-1.zip"],
        "filenames_url": ["https://zenodo.org/record/1492267/files/cvl-database-cropped-1-1.zip?download=1"],
        "filenames_sha256": ["b7e431dcfe3ea8e71b7df6a4da02548c97a78ceaec51b2d4cdf0f36b1f5bfe75"]
    }

    def __init__(self, num_classes_train, **kwargs):
        self.num_classes_train = num_classes_train
        super().__init__(**self._META_DATA, **kwargs)

    def _extract(self, src_file, outdir):
        _extract_zip(src_file, outdir)


class ICDAR2013Dataset(WriterRecognitionDataset):
    """ICDAR 2013 Dataset [1]

    References:
    [1] G. Louloudis, B. Gatos, N. Stamatopoulos, and A. Papandreou, ‘ICDAR 2013 Competition on Writer
    Identification’, in 2013 12th International Conference on Document Analysis and Recognition, Washington, DC, USA,
    Aug. 2013, pp. 1397–1401. doi: 10.1109/ICDAR.2013.282.
    """

    _META_DATA = {
        "identifier": "ICDAR2013",
        "filenames": ["experimental_dataset_2013.rar", "icdar2013_benchmarking_dataset.rar"],
        "filenames_url": [
            "https://users.iit.demokritos.gr/~louloud/ICDAR2013WriterIdentificationComp/experimental_dataset_2013.rar",
            "https://users.iit.demokritos.gr/~louloud/ICDAR2013WriterIdentificationComp"
            "/icdar2013_benchmarking_dataset.rar"],
        "filenames_sha256": ["3a3be2e8adeda5ca153888b692fad59cec20227a4237287d63437cce440a9e00",
                             "ed3a4bae3d691d50d66e370089b57991a61abb35cf0c7ed62c397c2af38a5056"]
    }

    def __init__(self, num_classes_train, **kwargs):
        self.num_classes_train = num_classes_train
        super().__init__(**self._META_DATA, **kwargs)

    def _extract(self, src_file, outdir):
        _extract_rar(src_file, outdir)


class WRITEDataset(WriterRecognitionDataset):
    """WRITE Dataset

    Version of the WRITE dataset, that has been cropped and segmented manually.
    This dataset is not publicly available.
    """

    _META_DATA = {
        "identifier": "WRITE cropped",
        "filenames": ["write-dataset-cropped-mko.zip"],
        "filenames_url": [None],  # not publicly available
        "filenames_sha256": ["1eb8eb319eadfd1c95503fb227165993f84e134e3591138ffd68294caabfaa44"]
    }

    def __init__(self, num_classes_train, **kwargs):
        self.num_classes_train = num_classes_train
        super().__init__(**self._META_DATA, **kwargs)

    def _extract(self, src_file, outdir):
        _extract_zip(src_file, outdir)


def _extract_zip(src_file, outdir):
    """
    Extracts the zip archive given by `src_file` to `outdir`.

    Args:
        src_file: Zip archive to be extracted
        outdir: Directory, where the content of `src_file` should be extracted to
    """
    with zipfile.ZipFile(src_file, "r") as archive:
        archive.extractall(outdir)


def _extract_rar(src_file, outdir):
    """
    Extracts the rar archive given by `src_file` to `outdir`.

    Args:
        src_file: Rar archive to be extracted
        outdir: Directory, where the content of `src_file` should be extracted to
    """
    dir = os.path.join(outdir, os.path.split(src_file)[1].split(".")[0])
    os.mkdir(dir)
    patoolib.extract_archive(src_file, outdir=dir)

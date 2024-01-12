import os

import cv2 as cv
import numpy as np
import pandas as pd
from src.exceptions import IllegalArgumentError
from src.datasets import CVLCroppedDataset, ICDAR2013Dataset, WRITEDataset

"""
Classes for preprocessing/transforming an instance of `WriterRetrievalDataset`
"""

this_file_dir = os.path.dirname(os.path.realpath(__file__))


class TransformationPipeline:
    """A pipeline consisting of single steps for transforming input images of a `WriterRetrievalDataset` instance"""

    dirname_preprocessed_files = "preprocessed"
    dataset_column_names = ["writer_id", "doc_id", "src_file", "set", "dir_dest"]

    def __init__(self, dataset_split_path, patch_extractor, dirname_out, pipeline_items=None,
                 conditional_filter=lambda img: False, quiet=False):
        """
        Args:
            dataset_split_path: Path to CSV file containing the dataset split
            patch_extractor: An instance of a class implementing all methods of `PatchExtractor`
            dirname_out: Name of the directory, where the preprocessed images should be stored (note: provide only the
            name of the immediate directory -- the root directory is provided by the respective
            `WriterRetrievalDataset` class when the preprocessing is started)
            pipeline_items (list, optional): A list of transformations steps to be applied to the patches extracted by
            `patch_extractor`; every item has to implement all methods of `TransformationPipelineItem`
            conditional_filter (optional): Function invoked after all steps of `pipeline_items` were executed on an
            input image. This function takes the image resulting from the preprocessing pipeline as an input and
            returns `True`, if the image should be filtered out (i.e. not saved), otherwise `False`.
            quiet (optional): Switches logging output during processing on/off
        """
        if pipeline_items is None:
            pipeline_items = []

        self.dataset_split = pd.read_csv(dataset_split_path, sep=";", dtype=str,
                                         names=self.dataset_column_names)
        self.patch_extractor = patch_extractor
        self.dirname_out = dirname_out
        self.pipeline_items = pipeline_items
        self.conditional_filter = conditional_filter
        self.quiet = quiet

    def __call__(self, root_dir, dirnames_extracted_datasets, *args, **kwargs):
        """"
        Args:
            root_dir: Path to the root directory, were the preprocessed images should be stored.
            dirnames_extracted_datasets (list): Paths to the directories containing the raw (unprocessed)
            extracted files
        """
        # create target directories in advance
        if not self._make_dirs(root_dir):
            return False

        num_images = len(self.dataset_split)

        for row in self.dataset_split.itertuples():
            self._log(
                f"Processing image {row.Index + 1}/{num_images} (writer id# {row.writer_id}, doc id# {row.doc_id})...")

            img_path = row.src_file

            for d in dirnames_extracted_datasets:
                img = cv.imread(os.path.join(root_dir, d, img_path))

                if img is not None and img.size != 0:
                    break
            else:
                assert False, f"No matching image for {img_path} found."

            self.patch_extractor(img)
            for patch_idx, img_patch in enumerate(self.patch_extractor):
                for transformation_step in self.pipeline_items:
                    img_patch = transformation_step(img_patch)

                # filter out patches according to filter condition
                if self.conditional_filter(img_patch):
                    continue

                self._save(
                    os.path.join(root_dir, self.dirname_preprocessed_files, self.dirname_out, row.set, row.dir_dest,
                                 f"{row.writer_id}-{row.doc_id}-{patch_idx}.jpg"), img_patch)

            self._log(f"[✅] Processed patches for image {row.Index + 1}/{num_images} successfully.")

        return True

    def _log(self, msg):
        if not self.quiet:
            print(msg)

    def _save(self, file_path, img):
        cv.imwrite(file_path, img)

    def _validate_dataset_split(self, dataset_split):
        raise NotImplementedError

    def _make_dirs(self, root_dir):
        # root directory for output
        try:
            os.makedirs(os.path.join(root_dir, self.dirname_preprocessed_files, self.dirname_out))
        except FileExistsError as e:
            return False

        # create sub-directories according to train/validation/test split
        columns_set_dir_dest = self.dataset_column_names[3:]
        # unique combinations of the set (train/validation/test) and the desired target directory
        dirs = self.dataset_split[columns_set_dir_dest].groupby(columns_set_dir_dest).groups.keys()

        for base_dir, sub_dir in dirs:
            os.makedirs(os.path.join(root_dir, self.dirname_preprocessed_files, self.dirname_out, base_dir, sub_dir))

        return True


class PatchExtractor:
    """An informal interface to be implemented by every patch extractor.

    A patch extractor is an iterable container, returning image patches from a given input image.

    Please refer to the methods, their documentation and their respective parameters in this class -- this is the
    minimum interface every patch extractor has to provide, i.e. those are the methods that need to be overwritten in
    a subclass.
    """

    def __init__(self):
        """
        This constructor only describes the interface and is not meant to be invoked by any subclass.
        """
        raise NotImplementedError

    def __call__(self, img, *args, **kwargs):
        """
        Method that sets the PatchExtractor into the state ready to return patches of `img`.

        Args:
            img: input image, where the patches are extracted from; format as returned by `cv.imread()`

        Returns:
            None
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """
        Returns the patch identified by `item`.

        Args:
            item: index of the patch

        Returns:
            the respective patch; format as returned by `cv.imread()`
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of image patches in this extractor.

        Returns:
            the number of image patches
        """
        raise NotImplementedError


class SIFTPatchExtractor(PatchExtractor):
    """Extracts image patches from an input image centered at SIFT key points [1].

    Reference:
    [1] D. G. Lowe, ‘Distinctive Image Features from Scale-Invariant Keypoints’, International Journal of Computer
    Vision, vol. 60, no. 2, pp. 91–110, Nov. 2004, doi: 10.1023/B:VISI.0000029664.99615.94.
    """

    def __init__(self, patch_size=(32, 32), num_features=0, num_octave_layers=3, contrast_threshold=0.04,
                 edge_threshold=10,
                 sigma=1.6):
        """
        Args:
            patch_size (optional): X and Y dimension of the patches to be extracted as tuple;
            each dimension has to divisible by 2
            num_features (optional): SIFT_create() parameter `nfeatures`, see [1]
            num_octave_layers (optional): SIFT_create() parameter `nOctaveLayers`, see [1]
            contrast_threshold (optional): SIFT_create() parameter `contrastThreshold`, see [1]
            edge_threshold (optional): SIFT_create() parameter `edgeThreshold`, see [1]
            sigma (optional): SIFT_create() parameter `sigma`, see [1]

        Reference:
        [1] https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1SIFT.html#ad337517bfdc068ae0ba0924ff1661131,
        Accessed: 2021-08-20
        """
        if patch_size[0] <= 0 or patch_size[0] % 2 or patch_size[1] <= 0 or patch_size[1] % 2:
            raise IllegalArgumentError

        self.patch_size = patch_size
        self.patch_size_div_2 = patch_size[0] // 2, patch_size[1] // 2
        self.sift_create_args = {"nfeatures": num_features, "nOctaveLayers": num_octave_layers,
                                 "contrastThreshold": contrast_threshold, "edgeThreshold": edge_threshold,
                                 "sigma": sigma}

    def __call__(self, img, *args, **kwargs):
        self.img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert image to grayscale, as needed by SIFT
        self.y_max, self.x_max, *_ = img.shape
        self.key_pts = self._detect_sift_key_points()
        self.n = len(self.key_pts)

    def __getitem__(self, item):
        if item < 0 or item >= self.n:
            raise IndexError

        return self._extract_patch(self.key_pts[item])

    def __len__(self):
        return self.n

    def _detect_sift_key_points(self):
        sift = cv.SIFT_create(**self.sift_create_args)
        return sift.detect(self.img, None)

    def _extract_patch(self, key_pt):
        center_x, center_y = int(key_pt.pt[0]), int(key_pt.pt[1])

        x1, x2, y1, y2 = center_x - self.patch_size_div_2[0], center_x + self.patch_size_div_2[0], center_y - \
                         self.patch_size_div_2[1], center_y + self.patch_size_div_2[1]

        # add white padding, in case patch exceeds image borders
        if y1 < 0 or y2 > self.y_max or x1 < 0 or x2 > self.x_max:
            top = abs(y1) if y1 < 0 else 0
            bottom = y2 - self.y_max if y2 > self.y_max else 0
            left = abs(x1) if x1 < 0 else 0
            right = x2 - self.x_max if x2 > self.x_max else 0

            img_patch = cv.copyMakeBorder(self.img[max(y1, 0):min(y2, self.y_max), max(x1, 0):min(x2, self.x_max)], top,
                                          bottom,
                                          left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            img_patch = self.img[y1:y2, x1:x2]

        assert img_patch.shape == self.patch_size, "Extracted patch size does not match desired target patch size"

        return img_patch


class TransformationPipelineItem:
    """An informal interface to be implemented by every transformation pipeline item.

    A transformation pipeline item is a callable container, returning a transformed version of an input image.

    Please refer to the methods, their documentation and their respective parameters in this class -- this is the
    minimum interface every transformation pipeline item has to provide, i.e. those are the methods that need to be
    overwritten in a subclass.
    """

    def __init__(self):
        """
        This constructor only describes the interface and is not meant to be invoked by any subclass.
        """
        raise NotImplementedError

    def __call__(self, img, *args, **kwargs):
        """
        Executes the pipeline step represented by this transformation pipeline item on `img`
        and returns the result.

        Args:
            img: image to be processed/transformed; format as returned by `cv.imread()`

        Returns:
            the transformed image; format as returned by `cv.imread()`
        """
        raise NotImplementedError


class OtsuBinarization(TransformationPipelineItem):
    """Binarises an image using Otsu's method [1].

    Reference:
    [1] N. Otsu, ‘A Threshold Selection Method from Gray-Level Histograms’, IEEE Trans. Syst., Man, Cybern., vol. 9,
    no. 1, pp. 62–66, Jan. 1979, doi: 10.1109/TSMC.1979.4310076.
    """

    def __init__(self):
        pass

    def __call__(self, img, *args, **kwargs):
        """
        Performs binarisation (Otsu's method) on `img` and returns the result.

        Args:
            img: image to be binarised; format as returned by `cv.imread()`

        Returns:
            the binarised image; format as returned by `cv.imread()`
        """
        _, img_binarized = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return img_binarized


# offers datasets incl. preprocessing used in our work
defaults = {
    "cvl-1-1_with-enrollment_pages": CVLCroppedDataset(
        310,
        transformation_pipeline=TransformationPipeline(
            os.path.join(this_file_dir, os.pardir, "dataset_splits", "cvl-1-1_with-enrollment_pages.csv"),
            SIFTPatchExtractor(sigma=3.75),
            "cvl-1-1_with-enrollment_pages",
            pipeline_items=[OtsuBinarization()]
        ), root_dir=os.path.join(os.curdir, "data")),
    # for retrieval-based evaluation -- writer with id 431 is excluded, since this writer has only one document in the
    # test set
    "cvl-1-1-test_retrieval-based-subset_with-enrollment_pages": CVLCroppedDataset(
        None,
        transformation_pipeline=TransformationPipeline(
            os.path.join(this_file_dir, os.pardir, "dataset_splits",
                         "cvl-1-1-test_retrieval-based-subset_with-enrollment_pages.csv"),
            SIFTPatchExtractor(sigma=3.75),
            "cvl-1-1-test_retrieval-based-subset_with-enrollment_pages",
            pipeline_items=[OtsuBinarization()]
        ), root_dir=os.path.join(os.curdir, "data")),
    "cvl-1-1_without-enrollment_pages": CVLCroppedDataset(
        27,
        transformation_pipeline=TransformationPipeline(
            os.path.join(this_file_dir, os.pardir, "dataset_splits", "cvl-1-1_without-enrollment_pages.csv"),
            SIFTPatchExtractor(sigma=3.75),
            "cvl-1-1_without-enrollment_pages",
            pipeline_items=[OtsuBinarization()],
            conditional_filter=lambda img: np.mean(img) == 255
        ), root_dir=os.path.join(os.curdir, "data")),
    "icdar-2013_pages": ICDAR2013Dataset(
        100,
        transformation_pipeline=TransformationPipeline(
            os.path.join(this_file_dir, os.pardir, "dataset_splits", "icdar-2013_pages.csv"),
            SIFTPatchExtractor(sigma=3.75),
            "icdar-2013_pages",
            pipeline_items=[OtsuBinarization()],
            # necessary, since there are many patches where nothing (or all most nothing) is visible
            conditional_filter=lambda img: np.sum(img == 255) > int(32 ** 2 * 0.95)
        ), root_dir=os.path.join(os.curdir, "data")),
    "icdar-2013-test_greek-subset_pages": ICDAR2013Dataset(
        None,
        transformation_pipeline=TransformationPipeline(
            os.path.join(this_file_dir, os.pardir, "dataset_splits", "icdar-2013-test_greek-subset_pages.csv"),
            SIFTPatchExtractor(sigma=3.75),
            "icdar-2013-test_greek-subset_pages",
            pipeline_items=[OtsuBinarization()],
            # necessary, since there are many patches where nothing (or all most nothing) is visible
            conditional_filter=lambda img: np.sum(img == 255) > int(32 ** 2 * 0.95)
        ), root_dir=os.path.join(os.curdir, "data")),
    "icdar-2013-test_latin-subset_pages": ICDAR2013Dataset(
        None,
        transformation_pipeline=TransformationPipeline(
            os.path.join(this_file_dir, os.pardir, "dataset_splits", "icdar-2013-test_latin-subset_pages.csv"),
            SIFTPatchExtractor(sigma=3.75),
            "icdar-2013-test_latin-subset_pages",
            pipeline_items=[OtsuBinarization()],
            # necessary, since there are many patches where nothing (or all most nothing) is visible
            conditional_filter=lambda img: np.sum(img == 255) > int(32 ** 2 * 0.95)
        ), root_dir=os.path.join(os.curdir, "data")),
}

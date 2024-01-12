import glob
import os
import random
import time

import numpy as np
from PIL import Image
import scipy.spatial.distance
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
# DO NOT REMOVE IMPORT
# workaround so tensorboard import does not fail (see https://github.com/pytorch/pytorch/pull/69904)
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

"""
Utility methods and classes for PyTorch training and inference for writer recognition tasks.
"""


def set_all_seeds(seed):
    """
    Ensures reproducible behaviour by resetting all seeds with the seed given by `seed`.
    Moreover, additional parameters are set to ensure deterministic behaviour.

    Reference:
    [1] https://pytorch.org/docs/stable/notes/randomness.html, Accessed: 2021-07-19

    Args:
        seed: The desired seed to be set
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Ensures reproducibility for `DataLoader` classes.

    This method is meant to be handed as an argument to the parameter `worker_init_fn` of a
    PyTorch `DataLoader`.

    Reference:
    [1] https://pytorch.org/docs/stable/notes/randomness.html#dataloader, Accessed: 2021-07-19

    Args:
        worker_id : Argument is handled by the respective `DataLoader`
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    """Class for training a model.

     This class supports also logging with `TensorBoard`"""

    def __init__(self, model, criterion, optimizer, scheduler, num_epochs, train_set_loader, val_set_loader,
                 experiment_name=None, hyper_params=None, num_epochs_early_stop=10, log_dir=None,
                 saved_models_dir=None):
        """
        Args:
            model: Model to be trained
            criterion: Desired criterion
            optimizer: Desired optimizer
            scheduler: Learning rate scheduler. Set this argument to `None`,
            if you do not want to use an LR scheduler
            num_epochs: Maximum number of epochs the model should be trained for
            train_set_loader: `DataLoader` instance of the training set
            val_set_loader: `DataLoader` instance of the validation set
            experiment_name (optional): Name of the experiment (has to be a valid name for
            a directory). If set to `None`, the experiment will be named 'experiment_<unix time stamp>'
            hyper_params (optional): Dictionary containing the hyper parameters of the trained model to be
            logged to `TensorBoard`
            num_epochs_early_stop (optional): Number of epochs after the training should be stopped,
            if the validation loss does not improve any more
            log_dir (optional): Path to the root directory, where the `TensorBoard` data should be logged to.
            If set to `None`, no logging takes place.
            saved_models_dir (optional): Path to the root directory, where the models should be saved to.
            A model is saved after each epoch, where the validation loss improved compared to the best so far
            obtained validation loss. If set to `None`, no models are saved.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.train_set_loader = train_set_loader
        self.val_set_loader = val_set_loader
        self.hyper_params = hyper_params
        self.num_epochs_early_stop = num_epochs_early_stop

        if experiment_name:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = "experiment_" + str(int(time.time() * 1000.0))

        self.log_path = None
        self.summary_writer = None
        if log_dir:
            self.log_path = os.path.join(log_dir, self.experiment_name)
            # ensures, that no previous experiment with the same name was already conducted in `log_dir`
            os.makedirs(self.log_path)
            self.summary_writer = SummaryWriter(self.log_path)

        self.saved_models_path = None
        if saved_models_dir:
            self.saved_models_path = os.path.join(saved_models_dir, self.experiment_name)
            os.makedirs(self.saved_models_path)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, *args, **kwargs):
        """Starts the training"""
        epoch_train_acc, epoch_val_acc, epoch_train_loss, epoch_val_loss = 0., 0., 0., 0.
        best_train_acc, best_val_acc, best_train_loss, best_val_loss = 0., 0., float('inf'), float('inf')

        early_stop_count = 0
        early_stop = False
        epoch = 0

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            if self.train_set_loader:
                epoch_train_acc, epoch_train_loss = self._train(epoch)

            if self.val_set_loader:
                epoch_val_acc, epoch_val_loss = self._validate()

            # logging
            if self.summary_writer:
                self.summary_writer.add_scalars("accuracy", {
                    "training": epoch_train_acc,
                    "validation": epoch_val_acc,
                }, epoch + 1)
                self.summary_writer.add_scalars("loss", {
                    "training": epoch_train_loss,
                    "validation": epoch_val_loss,
                }, epoch + 1)
                self.summary_writer.flush()

            if epoch_val_loss < best_val_loss:
                early_stop_count = 0

                if self.saved_models_path:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.saved_models_path, f"epoch_{epoch + 1}.pth"))
            else:
                early_stop_count += 1

            best_train_acc = (epoch_train_acc if epoch_train_acc > best_train_acc else best_train_acc)
            best_val_acc = (epoch_val_acc if epoch_val_acc > best_val_acc else best_val_acc)
            best_train_loss = (epoch_train_loss if epoch_train_loss < best_train_loss else best_train_loss)
            best_val_loss = (epoch_val_loss if epoch_val_loss < best_val_loss else best_val_loss)

            if early_stop_count == self.num_epochs_early_stop:
                print(f"Early stopping at epoch {epoch + 1} triggered.")
                early_stop = True
                break

        if self.summary_writer:
            if self.hyper_params:
                self.summary_writer.add_hparams(
                    self.hyper_params,
                    {
                        "hparams/acc_train": best_train_acc,
                        "hparams/acc_val": best_val_acc,
                        "hparams/loss_train": best_train_loss,
                        "hparams/loss_val": best_val_loss,
                        "hparams/num_epochs": epoch + 1 if not early_stop else epoch + 1 - self.num_epochs_early_stop
                    }
                )
            self.summary_writer.close()

    def _train(self, epoch):
        running_train_acc = 0
        running_train_loss = 0

        self.model.train()
        for data, label in tqdm(self.train_set_loader, total=len(self.train_set_loader)):
            data = data.to(device=self.device)
            label = label.to(device=self.device)

            output = self.model(data)
            loss = self.criterion(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_train_acc += (output.argmax(dim=1) == label).float().mean()
            running_train_loss += loss.item()

        if self.scheduler:
            self.scheduler.step()

        epoch_train_acc = running_train_acc / len(self.train_set_loader)
        epoch_train_loss = running_train_loss / len(self.train_set_loader)

        return epoch_train_acc, epoch_train_loss

    @torch.no_grad()
    def _validate(self):
        running_val_acc = 0
        running_val_loss = 0

        self.model.eval()
        for data, label in self.val_set_loader:
            data = data.to(device=self.device)
            label = label.to(device=self.device)

            output = self.model(data)
            loss = self.criterion(output, label)

            running_val_acc += (output.argmax(dim=1) == label).float().mean()
            running_val_loss += loss.item()

        epoch_val_acc = running_val_acc / len(self.val_set_loader)
        epoch_val_loss = running_val_loss / len(self.val_set_loader)

        return epoch_val_acc, epoch_val_loss


class ClassificationTester:
    """Class for testing a dataset as a classification task"""

    def __init__(self, test_set_path, model):
        """
        Args:
            test_set_path: Path to the preprocessed dataset to be tested
            model: Model to be used already set into evaluation mode and with
            loaded parameters (trained weights)
        """
        self.page_paths = glob.glob(os.path.join(test_set_path, "*/*"))
        self.model = model

        # get the same class to index mapping as in the training set based on `ImageFolder`
        self.class_to_idx = datasets.ImageFolder(os.path.join(test_set_path, os.pardir, "train")).class_to_idx
        self.num_classes = len(self.class_to_idx)

    @torch.no_grad()
    def __call__(self, device, batch_size, num_workers, top_k=None, *args, **kwargs):
        """Starts the classification-based evaluation

        Args:
            device: Device to be used (e.g. 'cuda')
            batch_size: Desired batch size (recommended: 1)
            num_workers: Number of PyTorch workers
            top_k (list, optional): Top k to be evaluated; each k should be given
            as a single entry in the list

        Returns:
            the evaluation result as a dictionary
        """
        if top_k is None:
            top_k = [1, 2, 3, 4, 5]
        top_k_correct = {k: 0 for k in top_k}

        seed = 417
        set_all_seeds(seed)

        for idx, page_path in enumerate(self.page_paths, 1):
            print(f"Testing page {idx}/{len(self.page_paths)}")

            page_class = page_path.split(os.sep)[-2]
            page_label = torch.tensor(self.class_to_idx[page_class])

            page = WriterItem(page_path, page_label, "jpg", transform=transforms.ToTensor())
            test_set_loader = DataLoader(dataset=page, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                         worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))

            output_avg = torch.zeros(self.num_classes).to(device=device)
            for data, _ in test_set_loader:  # label can be ignored, since one page has exactly one writer
                data = data.to(device=device)
                output = self.model(data)
                output_avg += output.mean(dim=0)

            output_avg /= len(test_set_loader)

            top_k = torch.topk(output_avg, max(top_k), dim=0).indices
            for k in top_k_correct.keys():
                top_k_correct[k] += torch.any(top_k[:k] == page_label).float().item()

        for k in top_k_correct.keys():
            top_k_correct[k] /= len(self.page_paths)

        return top_k_correct


class RetrievalTester:
    """Class for testing a dataset as a retrieval task

    Given a page of a writer divided into several image patches, the output of
    the transformer encoder and the MLP head of the network are used to form a global feature vector
    for the entire page by averaging them (similar to [1]).

    Besides the soft and hard criterion, also the mAP (mean average precision) is calculated.

    Reference:
    [1] S. Fiel and R. Sablatnig, ‘Writer Identification and Retrieval Using
    a Convolutional Neural Network’, in Computer Analysis of Images and Patterns,
    vol. 9257, G. Azzopardi and N. Petkov, Eds. Cham: Springer International Publishing, 2015, pp. 26–37.
    doi: 10.1007/978-3-319-23117-4_3.
    """

    def __init__(self, feature_vector_dims, test_set_path, model):
        """
        Args:
            feature_vector_dims (tuple): Dimension of the output of the transformer encoder and
            the MLP head of the network given as tuple (dim transformer encoder, dim mlp head)
            test_set_path: Path to the preprocessed dataset to be tested
            model: Model to be used already set into evaluation mode and with
            loaded parameters (trained weights)
        """
        self.page_paths = glob.glob(os.path.join(test_set_path, "*/*"))
        self.feature_vector_dims = feature_vector_dims
        self.model = model

        self.calculated_feature_vectors = False
        self.labels = None
        self.global_feature_vectors_transformer_encoder = None
        self.global_feature_vectors_mlp_head = None
        self.num_rel_docs_per_label = None

    @torch.no_grad()
    def __call__(self, device, batch_size, num_workers, soft_top_k=None, hard_top_k=None,
                 metrics=None, *args, **kwargs):
        """Starts the retrieval-based evaluation

        Args:
            device: Device to be used (e.g. 'cuda')
            batch_size: Desired batch size (recommended: 1)
            num_workers: Number of PyTorch workers
            soft_top_k (list, optional): Top k to be evaluated with the soft criterion;
            each k should be given as a single entry in the list
            hard_top_k (list, optional): Top k to be evaluated with the hard criterion;
            each k should be given as a single entry in the list
            metrics (list, optional): Distance metrics to be used for evaluation.
            Supported values see [1].

        Returns:
            the evaluation result as a dictionary

        Reference:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html,
        Accessed: 2021-09-16
        """
        if soft_top_k is None:
            soft_top_k = [1, 2, 3, 4, 5]
        if hard_top_k is None:
            hard_top_k = [1]
        if metrics is None:
            metrics = ["cosine"]

        if not self.calculated_feature_vectors:
            self.labels, self.global_feature_vectors_transformer_encoder, self.global_feature_vectors_mlp_head = \
                self._computer_feature_vectors(device, batch_size, num_workers)

            _, inv_idx, num_rel_docs_inv = self.labels.unique(return_inverse=True, return_counts=True)
            self.num_rel_docs_per_label = (num_rel_docs_inv[inv_idx] - 1)
            self.calculated_feature_vectors = True

        assert self.labels is not None and self.global_feature_vectors_transformer_encoder is not None and \
               self.global_feature_vectors_mlp_head is not None and self.num_rel_docs_per_label is not None, \
            "Feature vectors were not calculated"

        assert torch.any(
            self.num_rel_docs_per_label > 0), "Cannot perform retrieval-based evaluation: There is a writer with " \
                                              "only one document in the test set"

        return {"transformer_encoder": self._evaluate(self.global_feature_vectors_transformer_encoder, self.labels,
                                                      self.num_rel_docs_per_label, soft_top_k, hard_top_k, metrics),
                "mlp_head": self._evaluate(self.global_feature_vectors_mlp_head, self.labels,
                                           self.num_rel_docs_per_label,
                                           soft_top_k, hard_top_k, metrics)}

    @torch.no_grad()
    def _computer_feature_vectors(self, device, batch_size, num_workers):
        seed = 417
        set_all_seeds(seed)

        results_intermediate_layers = {}
        hook_transformer_encoder = self.model.to_latent.register_forward_hook(
            self._get_intermediate_layer(results_intermediate_layers, "to_latent"))

        global_feature_vectors_transformer_encoder = torch.zeros((len(self.page_paths), self.feature_vector_dims[0])).to(
            device=device)
        global_feature_vectors_mlp_head = torch.zeros((len(self.page_paths), self.feature_vector_dims[1])).to(
            device=device)
        labels = torch.zeros((len(self.page_paths),), dtype=torch.int)

        for idx, page_path in enumerate(self.page_paths, 0):
            print(f"Calculating feature vector for page {idx + 1}/{len(self.page_paths)}")

            page_label = page_path.split(os.sep)[-2]
            page = WriterItem(page_path, page_label, "jpg", transform=transforms.ToTensor())
            test_set_loader = DataLoader(dataset=page, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                         worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))

            for num_batches, (data, label) in enumerate(test_set_loader, 1):
                data = data.to(device=device)
                output = self.model(data)

                global_feature_vectors_transformer_encoder[idx] += results_intermediate_layers["to_latent"].mean(dim=0)
                global_feature_vectors_mlp_head[idx] += output.mean(dim=0)

            global_feature_vectors_transformer_encoder[idx] /= num_batches
            global_feature_vectors_mlp_head[idx] /= num_batches
            labels[idx] = float(label[0])

        hook_transformer_encoder.remove()

        return labels, global_feature_vectors_transformer_encoder, global_feature_vectors_mlp_head

    @staticmethod
    def _get_intermediate_layer(activations, key):
        def hook(model, input, output):
            activations[key] = output

        return hook

    @staticmethod
    def _evaluate(global_feature_vectors, labels, num_rel_docs_per_label, soft_top_k, hard_top_k, metrics):
        global_feature_vectors_norm = preprocessing.normalize(global_feature_vectors.detach().cpu().numpy())

        result = {}
        for m in metrics:
            result[m] = {}
            # dist_matrix rows (dim 0): distance to other documents
            # dist_matrix columns (dim 1): query documents
            # the distance on the diagonal is set to infinity, since the distance of a query document
            # to itself should not be considered
            dist_matrix = torch.from_numpy(scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(global_feature_vectors_norm, metric=m))).float().fill_diagonal_(
                float("Inf"))

            num_docs = dist_matrix.shape[0]
            ranking = torch.topk(dist_matrix, num_docs, dim=0, largest=False).indices

            soft_top_k_result = {}
            for k in soft_top_k:
                soft_top_k_result[k] = (labels[ranking[:k]] == labels).any(dim=0).float().mean().item()

            hard_top_k_result = {}
            for k in hard_top_k:
                hard_top_k_result[k] = (labels[ranking[:k]] == labels).all(dim=0).float().mean().item()

            result[m]["soft_top_k"] = soft_top_k_result
            result[m]["hard_top_k"] = hard_top_k_result

            # mAP
            prec_at_k = (labels[ranking] == labels).float().cumsum(dim=0) / torch.arange(1, num_docs + 1).unsqueeze(
                0).t()
            rel_k = (labels[ranking] == labels).float()  # mask for filtering the relevant documents
            ap = (prec_at_k * rel_k).sum(dim=0) / num_rel_docs_per_label
            result[m]["mAP"] = (ap.sum() / num_docs).item()

        return result


class WriterItem(torch.utils.data.Dataset):
    """Custom PyTorch Dataset representing a single handwritten page image
    that can consist of multiple image patches (as extracted during preprocessing)"""

    def __init__(self, img_dir, label, img_extension="jpg", transform=None):
        """
        Args:
            img_dir: Directory containing the page image or image patches
            label: Respective label
            img_extension (optional): File extension/type of the images
            transform (optional): Transformation to be applied
        """
        self.img_dir = img_dir
        self.label = label
        self.img_extension = img_extension
        self.transform = transform
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*." + self.img_extension)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        # expand single channel images to three channels (needed by model)
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        return img, self.label

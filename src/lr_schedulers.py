class WarmUpLR:
    """Learning rate warmup scheduler as used by [1] (implementation: [2]).

    The learning rate is warmed up by dividing the initial learning rate by (`num_epochs_warm_up` - i),
    where i is the current epoch (0 based indexing). After the learning rate was warmed up for `num_epochs_warm_up`
    epochs,
    it stays the same.

    Example: `num_epochs_warm_up` = 5, lr ... learning rate
    1. Epoch: lr = initial lr / (5 - 0) = initial lr / 5
    2. Epoch: lr = initial lr / (5 - 1) = initial lr / 4
    3. Epoch: lr = initial lr / (5 - 2) = initial lr / 3
    4. Epoch: lr = initial lr / (5 - 3) = initial lr / 2
    >= 5. Epoch: lr = initial lr / (5 - 4) = initial lr / 1 = initial lr

    Note: A warm up learning rate scheduler will be implemented in PyTorch in a future release (see [3]).

    References:
    [1] A. Hassani, S. Walton, N. Shah, A. Abuduweili, J. Li, and H. Shi, ‘Escaping the Big Data Paradigm with
    Compact Transformers’, arXiv:2104.05704 [cs], Jun. 2021, Accessed: Jul. 19, 2021. [Online]. Available:
    http://arxiv.org/abs/2104.05704
    [2] https://github.com/SHI-Labs/Compact-Transformers/blob/e7fe3532dd17c4dafd5afae32082e96c4bf780b3/main.py#L186,
    Accessed: 2021-08-22
    [3] https://github.com/pytorch/pytorch/pull/60836, Accessed: 2021-08-22
    """

    def __init__(self, optimizer, initial_lr, num_epochs_warm_up=10):
        """
        Args:
            optimizer: The used optimizer
            initial_lr: The initial learning rate
            num_epochs_warm_up (optional): The number of epochs the learning rate should be warmed up
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.num_epochs_warm_up = num_epochs_warm_up
        self.last_epoch = 0

    def step(self):
        if self.last_epoch < self.num_epochs_warm_up:
            lr = self.initial_lr / (self.num_epochs_warm_up - self.last_epoch)
        else:
            lr = self.initial_lr

        self.last_epoch += 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

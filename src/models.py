from vit_pytorch import ViT

"""
Predefined ViT models.
"""


def vit_lite_7_4(num_classes):
    """
    The ViT-Lite-7/4 as proposed by Hassani et al [1].

    Reference:
    [1] A. Hassani, S. Walton, N. Shah, A. Abuduweili, J. Li, and H. Shi, ‘Escaping the Big Data Paradigm with Compact
    Transformers’, arXiv:2104.05704 [cs], Jun. 2021, Accessed: Jul. 19, 2021. [Online]. Available:
    http://arxiv.org/abs/2104.05704

    Args:
        num_classes: Number of classes

    Returns:
        A ViT-Lite-7/4 instance
    """
    return ViT(
        image_size=32,
        patch_size=4,
        num_classes=num_classes,
        dim=256,
        depth=7,
        heads=4,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.
    )

# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.registry import DATASETS
from mmpretrain.registry import PIPELINES


def build_dataset(cfg):
    """Build dataset.

    Examples:
        >>> from mmpretrain.datasets import build_dataset
        >>> mnist_train = build_dataset(
        ...     dict(type='MNIST', data_prefix='data/mnist/', test_mode=False))
        >>> print(mnist_train)
        Dataset MNIST
            Number of samples:  60000
            Number of categories:       10
            Prefix of data:     data/mnist/
        >>> mnist_test = build_dataset(
        ...     dict(type='MNIST', data_prefix='data/mnist/', test_mode=True))
        >>> print(mnist_test)
        Dataset MNIST
            Number of samples:  10000
            Number of categories:       10
            Prefix of data:     data/mnist/
    """
    return DATASETS.build(cfg)

def build_pipeline(cfg):
    """Build pipeline.
    
    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        pipeline: Constructed pipeline.
    """
    return PIPELINES.build(cfg)

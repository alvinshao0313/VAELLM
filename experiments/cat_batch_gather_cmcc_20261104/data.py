"""Vectorize CPU gathering without changing the native sampler or its RNG."""
import torch

from train_utils.cat_train_data import build_block_data_loaders as native_loaders


class _BatchDataset(torch.utils.data.TensorDataset):
    def __getitems__(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long, device='cpu')
        return [tensor.index_select(0, index) for tensor in self.tensors]


def _already_batched(batch):
    return batch


def _batch_gather(loader):
    if loader.num_workers != 0 or loader.pin_memory:
        raise ValueError('This adapter preserves the native single-process, unpinned CPU path.')
    dataset = loader.dataset
    if type(dataset) is not torch.utils.data.TensorDataset:
        raise ValueError('Expected the actual native TensorDataset.')
    if any(tensor.device.type != 'cpu' for tensor in dataset.tensors):
        raise ValueError('CPU batch gathering requires CPU source tensors.')
    return torch.utils.data.DataLoader(
        _BatchDataset(*dataset.tensors),
        batch_sampler=loader.batch_sampler,
        generator=loader.generator,
        collate_fn=_already_batched,
        num_workers=0,
        pin_memory=False,
    )


def build_block_data_loaders(stacked_data, batch_size, shuffle_seed=None):
    train, evaluation = native_loaders(stacked_data, batch_size, shuffle_seed)
    return _batch_gather(train), _batch_gather(evaluation)

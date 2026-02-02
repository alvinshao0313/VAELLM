import torch
import torch.nn.functional as F


class WeightMatrixDataset(torch.utils.data.Dataset):
    """
    一个简单的 Dataset，用于将整个权重矩阵切分为 chunk_size 大小的向量。
    """

    def __init__(self, matrix, chunk_size=32):
        super().__init__()
        self.matrix = matrix
        self.chunk_size = chunk_size

        # 将矩阵打平并切分为 [N, chunk_size]
        flat = self.matrix.flatten()
        if flat.numel() % chunk_size != 0:
            pad_len = chunk_size - (flat.numel() % chunk_size)
            flat = F.pad(flat, (0, pad_len))

        self.chunks = flat.reshape(-1, chunk_size)

    def __len__(self):
        return self.chunks.shape[0]

    def __getitem__(self, idx):
        # 返回一个向量 [chunk_size]
        return {
            "weight": self.chunks[idx],
            "index": idx
        }

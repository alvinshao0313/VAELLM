import argparse
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from sklearn.cluster import MiniBatchKMeans
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--codebook_bits", type=int, default=16, help="Log2 of number of clusters")
    parser.add_argument("--codebook_dim", type=int, default=8, help="Dimension of each vector (chunk size)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4096 * 10, help="Batch size for MiniBatchKMeans")
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Loading model from {args.model_path}...")
    try:
        # Load logic consistent with train_llm.py
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu"
        )

        # Hardcoded layer selection as per train_llm.py current state
        # matrix = llm.model.layers[0].self_attn.q_proj.weight.detach().float()
        # To make it robust if user changes args in future, we stick to the hardcoded request:
        matrix = llm.model.layers[0].self_attn.q_proj.weight.detach().float()

        logger.info(f"Loaded weight matrix: Layer 0, Module q_proj, Shape {matrix.shape}")

        del llm
        import gc
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to load matrix from model: {e}")
        raise e

    # Data preparation
    chunk_size = args.codebook_dim
    num_elements = matrix.numel()

    # Handle padding if necessary (though 4096*4096 is likely divisible)
    if num_elements % chunk_size != 0:
        pad_len = chunk_size - (num_elements % chunk_size)
        flat_matrix = torch.nn.functional.pad(matrix.flatten(), (0, pad_len))
    else:
        flat_matrix = matrix.flatten()

    X = flat_matrix.reshape(-1, chunk_size).numpy()
    logger.info(f"Data shape for K-Means: {X.shape}")

    n_clusters = 2 ** args.codebook_bits
    logger.info(f"Running K-Means with {n_clusters} clusters (bits={args.codebook_bits})...")
    logger.info("Using MiniBatchKMeans for efficiency.")

    start_time = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=args.batch_size,
        random_state=args.seed,
        n_init='auto',
        max_iter=args.max_iter,
        verbose=1
    )

    # Fit
    kmeans.fit(X)
    fit_time = time.time() - start_time
    logger.info(f"K-Means fit finished in {fit_time:.2f} seconds.")

    # Reconstruct
    logger.info("Reconstructing...")
    cluster_centers = kmeans.cluster_centers_  # (K, D)
    labels = kmeans.labels_  # (N,) of ints

    # Replace each vector with its centroid
    X_recon = cluster_centers[labels]

    # Calculate MSE
    mse = np.mean((X - X_recon) ** 2)
    logger.info(f"===> Full Matrix MSE (K-Means Baseline): {mse:.6e}")


if __name__ == "__main__":
    main()

from sparse_bit_tuning.triton_kernels import _q_bank_grid


def test_large_sparse_bit_q_dimension_is_mapped_to_grid_x():
    # Regression for CUDA grid.y <= 65535: a realistic down_proj bank at
    # active_ratio=0.03 needs more than 65535 dscore Q tiles when BLOCK_Q=32.
    max_active = 2_956_985
    grid = _q_bank_grid(max_active=max_active, block=32, num_banks=2)

    assert grid == (92_406, 2)
    assert grid[0] > 65_535
    assert grid[1] == 2

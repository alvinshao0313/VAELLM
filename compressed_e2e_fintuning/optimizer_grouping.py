"""E2E thin adapter over shared model-level optimizer grouping."""

from train_utils.model_level_optimizer import create_model_level_optimizer


def create_decoder_grouped_optimizer(trainer):
    """Create continuous E2E optimizer via shared inventory-based grouping.

    Requires ``trainer.model_level_trainable_selection`` (Task 6 inventories).
    """
    return create_model_level_optimizer(trainer)

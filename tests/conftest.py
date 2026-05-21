import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    return tmp_path / "checkpoint"

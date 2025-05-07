import torch
from torch.testing import assert_close
from nocap.models import make_causal_mask

def test_make_causal_mask():
    prefix_len = 2  # vision tokens
    sequence_len = 3  # text tokens
    device = torch.device('cpu')

    mask = make_causal_mask(prefix_len, sequence_len, device)

    expected_mask = torch.tensor([
        [False, False, True, True, True],  # Vision token 1
        [False, False, True, True, True],  # Vision token 2
        [False,  False, False, True, True],   # Text token 1 can't see future
        [False,  False, False,  False, True],   # Text token 2
        [False,  False, False,  False,  False],   # Text token 3
    ], dtype=torch.bool)

    
    assert mask.shape == (5, 5)
    assert_close(mask, expected_mask)

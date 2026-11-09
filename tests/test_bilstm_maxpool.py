"""Tests for the bilstm_maxpool architecture (Day 4 Step 2)."""
import torch
import pytest

from ml.models import Verifier, masked_max


def test_bilstm_maxpool_output_shape():
    """bilstm_maxpool produces correct output shape (batch, 3 classes)."""
    torch.manual_seed(42)
    model = Verifier(arch="bilstm_maxpool", vocab_size=50, emb_dim=32, hidden=64, dropout=0.0)
    model.eval()
    claim = torch.tensor([[1, 5, 8, 2, 0], [3, 4, 0, 0, 0]])
    ev = torch.tensor([[2, 4, 6, 8, 1, 3], [5, 7, 9, 0, 0, 0]])
    claim_mask = claim != 0
    ev_mask = ev != 0
    with torch.no_grad():
        logits = model(claim, claim_mask, ev, ev_mask)
    assert logits.shape == (2, 3)


def test_bilstm_maxpool_padding_invariance():
    """Extra padding must not change the max-pooled result."""
    torch.manual_seed(42)
    model = Verifier(arch="bilstm_maxpool", vocab_size=50, emb_dim=32, hidden=64, dropout=0.0)
    model.eval()
    # Batch with no extra padding
    claim1 = torch.tensor([[1, 5, 8]])
    ev1 = torch.tensor([[2, 4, 6]])
    cm1 = claim1 != 0
    em1 = ev1 != 0
    # Same content but with extra padding
    claim2 = torch.tensor([[1, 5, 8, 0, 0]])
    ev2 = torch.tensor([[2, 4, 6, 0, 0]])
    cm2 = claim2 != 0
    em2 = ev2 != 0
    with torch.no_grad():
        logits1 = model(claim1, cm1, ev1, em1)
        logits2 = model(claim2, cm2, ev2, em2)
    torch.testing.assert_close(logits1, logits2, atol=1e-5, rtol=1e-5)


def test_bilstm_maxpool_param_count_equals_bilstm():
    """bilstm_maxpool must have the same parameter count as bilstm (no extra params)."""
    torch.manual_seed(42)
    model_bilstm = Verifier(arch="bilstm", vocab_size=100, emb_dim=100, hidden=128, dropout=0.3)
    model_maxpool = Verifier(arch="bilstm_maxpool", vocab_size=100, emb_dim=100, hidden=128, dropout=0.3)
    n_bilstm = sum(p.numel() for p in model_bilstm.parameters())
    n_maxpool = sum(p.numel() for p in model_maxpool.parameters())
    assert n_bilstm == n_maxpool, f"bilstm={n_bilstm} vs bilstm_maxpool={n_maxpool}"


def test_masked_max_function():
    """masked_max ignores padded positions and selects the correct maximum."""
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[5.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    mask = torch.tensor([
        [True, True, False],
        [True, False, False],
    ])
    result = masked_max(x, mask)
    # Batch 0: max over positions 0 and 1 -> [3.0, 4.0]
    # Batch 1: max over position 0 only -> [5.0, 1.0]
    expected = torch.tensor([[3.0, 4.0], [5.0, 1.0]])
    torch.testing.assert_close(result, expected)

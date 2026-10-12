"""Tests for ``syrinx.generate.generate``. A table model makes the expected tokens exact."""

import pytest
import torch
from torch import nn

from syrinx.config import ModelConfig
from syrinx.generate import generate
from syrinx.model import SyrinxLM

V = 8


class TableModel(nn.Module):
    """Logits at position t depend only on token t: ``table[ids]``. Records input shapes."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table.clone(), requires_grad=False)
        self.seen: list[tuple[int, ...]] = []
        self.contexts: list[torch.Tensor] = []
        self.modes: list[bool] = []

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.seen.append(tuple(ids.shape))
        self.contexts.append(ids.clone())
        self.modes.append(self.training)
        return self.table[ids]


def _chain() -> torch.Tensor:
    """Greedy successor of token i is (i + 1) % V, with a clear margin."""
    table = torch.zeros(V, V)
    for i in range(V):
        table[i, (i + 1) % V] = 5.0
        table[i, (i + 2) % V] = 4.0
        table[i, (i + 3) % V] = 3.0
    return table


def test_greedy_follows_argmax() -> None:
    model = TableModel(_chain())
    out = generate(model, torch.tensor([[2]]), max_new_tokens=5, max_seq_len=16, temperature=0)
    assert out.tolist() == [[2, 3, 4, 5, 6, 7]]
    assert out.dtype == torch.int64


def test_prompt_is_kept_and_batched() -> None:
    model = TableModel(_chain())
    prompt = torch.tensor([[0, 1], [6, 7]])
    out = generate(model, prompt, max_new_tokens=3, max_seq_len=16, temperature=0)
    assert out.shape == (2, 5)
    assert torch.equal(out[:, :2], prompt)
    assert out[:, 2:].tolist() == [[2, 3, 4], [0, 1, 2]]


def test_uses_last_position_logits() -> None:
    """Next token depends on the LAST token of the context, not the first."""
    model = TableModel(_chain())
    out = generate(model, torch.tensor([[5, 0]]), max_new_tokens=1, max_seq_len=16, temperature=0)
    assert out[0, -1].item() == 1


def test_zero_new_tokens_returns_prompt() -> None:
    model = TableModel(_chain())
    prompt = torch.tensor([[3, 4]])
    assert torch.equal(generate(model, prompt, 0, 16), prompt)
    assert model.seen == []


def test_context_is_cropped_to_max_seq_len() -> None:
    model = TableModel(_chain())
    out = generate(
        model, torch.tensor([[0, 1, 2, 3, 4, 5]]), max_new_tokens=4, max_seq_len=4, temperature=0
    )
    assert out.shape == (1, 10)
    assert all(shape[1] <= 4 for shape in model.seen)
    assert model.seen[0] == (1, 4)
    # each step sees the NEWEST max_seq_len tokens generated so far
    for i, context in enumerate(model.contexts):
        assert torch.equal(context, out[:, 2 + i : 6 + i])


def test_top_k_one_equals_greedy() -> None:
    model = TableModel(_chain())
    gen = torch.Generator().manual_seed(0)
    out = generate(model, torch.tensor([[2]]), 6, 16, temperature=1.0, top_k=1, generator=gen)
    assert out.tolist() == [[2, 3, 4, 5, 6, 7, 0]]


def test_top_k_never_samples_outside_k() -> None:
    model = TableModel(_chain())
    gen = torch.Generator().manual_seed(1)
    prompt = torch.zeros(64, 1, dtype=torch.long)
    out = generate(model, prompt, 1, 16, temperature=5.0, top_k=2, generator=gen)
    assert set(out[:, 1].tolist()) <= {1, 2}
    assert set(out[:, 1].tolist()) == {1, 2}  # high temperature: both appear


def test_sampling_matches_distribution() -> None:
    """With temperature T the sampled frequencies follow softmax(logits / T)."""
    table = torch.zeros(V, V)
    table[0, :3] = torch.tensor([2.0, 1.0, 0.0])
    table[0, 3:] = -1e9
    model = TableModel(table)
    gen = torch.Generator().manual_seed(2)
    out = generate(
        model, torch.zeros(20_000, 1, dtype=torch.long), 1, 16, temperature=2.0, generator=gen
    )
    freq = torch.bincount(out[:, 1], minlength=V).float() / 20_000
    want = torch.softmax(torch.tensor([2.0, 1.0, 0.0]) / 2.0, dim=0)
    torch.testing.assert_close(freq[:3], want, atol=0.015, rtol=0)
    assert freq[3:].sum() == 0


def test_seeded_sampling_is_reproducible() -> None:
    model = TableModel(torch.randn(V, V, generator=torch.Generator().manual_seed(3)))
    runs = [
        generate(
            model,
            torch.tensor([[1]]),
            20,
            16,
            temperature=1.0,
            generator=torch.Generator().manual_seed(9),
        )
        for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])


def test_eos_stops_and_pads() -> None:
    model = TableModel(_chain())
    # row 0 reaches EOS=5 after 2 tokens, row 1 after 6
    out = generate(
        model, torch.tensor([[3], [7]]), max_new_tokens=10, max_seq_len=16, temperature=0, eos_id=5
    )
    assert out.tolist() == [[3, 4, 5, 5, 5, 5, 5], [7, 0, 1, 2, 3, 4, 5]]


def test_eos_single_row_stops_early() -> None:
    model = TableModel(_chain())
    out = generate(model, torch.tensor([[3]]), 50, 16, temperature=0, eos_id=5)
    assert out.tolist() == [[3, 4, 5]]


def test_eval_mode_during_and_restored_after() -> None:
    model = TableModel(_chain()).train()
    generate(model, torch.tensor([[1]]), 3, 16, temperature=0)
    assert model.modes == [False, False, False]
    assert model.training
    model.eval()
    generate(model, torch.tensor([[1]]), 1, 16, temperature=0)
    assert not model.training


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": torch.tensor([1, 2])},
        {"prompt": torch.zeros(1, 0, dtype=torch.long)},
        {"max_new_tokens": -1},
        {"temperature": -0.5},
        {"top_k": 0},
    ],
)
def test_invalid_arguments(kwargs: dict) -> None:  # type: ignore[type-arg]
    args = {"prompt": torch.tensor([[1]]), "max_new_tokens": 2, "max_seq_len": 16}
    args.update(kwargs)
    with pytest.raises(ValueError):
        generate(TableModel(_chain()), **args)


def test_real_model_on_device(each_device: torch.device) -> None:
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=96,
        max_seq_len=8,
        rope_theta=10000.0,
        norm_eps=1e-5,
    )
    torch.manual_seed(0)
    model = SyrinxLM(cfg).to(each_device)
    prompt = torch.tensor([[1, 2, 3]], device=each_device)
    greedy = generate(model, prompt, 12, cfg.max_seq_len, temperature=0)
    assert greedy.shape == (1, 15) and greedy.device.type == each_device.type
    assert torch.equal(greedy, generate(model, prompt, 12, cfg.max_seq_len, temperature=0))
    gen = torch.Generator(device=each_device).manual_seed(0)
    sampled = generate(model, prompt, 12, cfg.max_seq_len, top_k=10, generator=gen)
    assert sampled.shape == (1, 15) and int(sampled.max()) < 64

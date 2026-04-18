"""Smoke test: build a tiny pretrain model and run one forward + loss pass."""
import pytest
import torch


@pytest.fixture
def tiny_cfg():
    from uni_react.configs import PretrainConfig
    return PretrainConfig(
        emb_dim=32,
        inv_layer=1,
        se3_layer=1,
        heads=4,
        atom_vocab_size=16,
        mask_token_id=15,
        cutoff=3.0,
        num_kernel=16,
        train_mode="geometric_structure",
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    from uni_react.encoders import SingleMolPretrainNet
    return SingleMolPretrainNet(
        emb_dim=tiny_cfg.emb_dim,
        inv_layer=tiny_cfg.inv_layer,
        se3_layer=tiny_cfg.se3_layer,
        heads=tiny_cfg.heads,
        atom_vocab_size=tiny_cfg.atom_vocab_size,
        cutoff=tiny_cfg.cutoff,
        num_kernel=tiny_cfg.num_kernel,
    )


@pytest.fixture
def tiny_batch():
    B, N = 2, 6
    torch.manual_seed(0)
    return {
        "input_atomic_numbers": torch.randint(1, 8, (B, N)),
        "atomic_numbers":       torch.randint(1, 8, (B, N)),
        "coords_noisy":         torch.randn(B, N, 3),
        "coords":               torch.randn(B, N, 3),
        "atom_padding":         torch.zeros(B, N, dtype=torch.bool),
        "mask_positions":       torch.zeros(B, N, dtype=torch.bool),
        "charges":              torch.randn(B, N),
    }


def test_forward_runs(tiny_model, tiny_batch):
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(
            input_atomic_numbers=tiny_batch["input_atomic_numbers"],
            coords_noisy=tiny_batch["coords_noisy"],
            atom_padding=tiny_batch["atom_padding"],
        )
    assert "node_feats" in out
    assert "graph_feats" in out
    assert out["node_feats"].shape[:2] == (2, 6)


def test_geometric_loss(tiny_model, tiny_batch):
    from uni_react.losses import GeometricStructureLoss
    loss_fn = GeometricStructureLoss()
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(
            input_atomic_numbers=tiny_batch["input_atomic_numbers"],
            coords_noisy=tiny_batch["coords_noisy"],
            atom_padding=tiny_batch["atom_padding"],
        )
        losses = loss_fn(out, tiny_batch)
    assert "loss" in losses
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["atom_loss"])
    assert torch.isfinite(losses["coord_loss"])
    assert torch.isfinite(losses["charge_loss"])


def test_geometric_loss_uses_dataset_style_batch_keys(tiny_model):
    from uni_react.losses import GeometricStructureLoss

    torch.manual_seed(0)
    batch = {
        "input_atomic_numbers": torch.tensor([[15, 6, 7, 8]], dtype=torch.long),
        "atomic_numbers": torch.tensor([[1, 6, 7, 8]], dtype=torch.long),
        "coords_noisy": torch.randn(1, 4, 3),
        "coords": torch.randn(1, 4, 3),
        "atom_padding": torch.zeros(1, 4, dtype=torch.bool),
        "mask_positions": torch.tensor([[True, False, False, False]], dtype=torch.bool),
        "charges": torch.randn(1, 4),
        "charge_valid": torch.ones(1, 4, dtype=torch.bool),
    }
    loss_fn = GeometricStructureLoss()

    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(
            input_atomic_numbers=batch["input_atomic_numbers"],
            coords_noisy=batch["coords_noisy"],
            atom_padding=batch["atom_padding"],
        )
        losses = loss_fn(out, batch)

    assert float(losses["atom_loss"]) > 0.0
    assert float(losses["coord_loss"]) > 0.0
    assert float(losses["charge_loss"]) >= 0.0


def test_backward_runs(tiny_model, tiny_batch):
    from uni_react.losses import GeometricStructureLoss
    loss_fn = GeometricStructureLoss()
    tiny_model.train()
    out = tiny_model(
        input_atomic_numbers=tiny_batch["input_atomic_numbers"],
        coords_noisy=tiny_batch["coords_noisy"],
        atom_padding=tiny_batch["atom_padding"],
    )
    losses = loss_fn(out, tiny_batch)
    losses["loss"].backward()
    # Check at least one parameter received a gradient
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_lidi_forward_and_loss_smoke(tiny_cfg):
    from uni_react.encoders import SingleMolPretrainNet
    from uni_react.losses import GeometricStructureLoss

    torch.manual_seed(0)
    model = SingleMolPretrainNet(
        emb_dim=tiny_cfg.emb_dim,
        inv_layer=tiny_cfg.inv_layer,
        se3_layer=tiny_cfg.se3_layer,
        heads=tiny_cfg.heads,
        atom_vocab_size=tiny_cfg.atom_vocab_size,
        cutoff=tiny_cfg.cutoff,
        num_kernel=tiny_cfg.num_kernel,
        enable_lidi_task=True,
        geometric_lidi_hidden_dim=64,
    )
    batch = {
        "input_atomic_numbers": torch.tensor([[15, 6, 7, 8]], dtype=torch.long),
        "atomic_numbers": torch.tensor([[1, 6, 7, 8]], dtype=torch.long),
        "coords_noisy": torch.randn(1, 4, 3),
        "coords": torch.randn(1, 4, 3),
        "atom_padding": torch.zeros(1, 4, dtype=torch.bool),
        "mask_positions": torch.tensor([[True, False, False, False]], dtype=torch.bool),
        "charges": torch.randn(1, 4),
        "charge_valid": torch.ones(1, 4, dtype=torch.bool),
        "lidi_matrix": torch.randn(1, 4, 4),
        "lidi_valid": torch.ones(1, 4, 4, dtype=torch.bool),
    }
    # Make target LIDI symmetric for a realistic signal.
    batch["lidi_matrix"] = 0.5 * (batch["lidi_matrix"] + batch["lidi_matrix"].transpose(1, 2))

    loss_fn = GeometricStructureLoss(
        atom_weight=1.0,
        coord_weight=1.0,
        charge_weight=0.0,
        lidi_node_weight=1.0,
        lidi_edge_weight=0.2,
        lidi_conservation_weight=0.1,
    )

    model.eval()
    with torch.no_grad():
        out = model(
            input_atomic_numbers=batch["input_atomic_numbers"],
            coords_noisy=batch["coords_noisy"],
            atom_padding=batch["atom_padding"],
            active_pipeline_tasks=("geometric_structure",),
            active_geometric_subtasks=("atom_mask", "coord_denoise", "lidi"),
        )
        losses = loss_fn(out, batch)

    assert "lidi_pred" in out
    assert out["lidi_pred"].shape == (1, 4, 4)
    assert torch.isfinite(losses["lidi_node_loss"])
    assert torch.isfinite(losses["lidi_edge_loss"])
    assert torch.isfinite(losses["lidi_conservation_loss"])

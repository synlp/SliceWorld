import torch

from sliceworld.models.losses import SliceWorldObjective, loss_config_from_dict
from sliceworld.models.sliceworld import SliceWorldCore, SliceWorldForCTRG


def smoke_config():
    return {
        "dropout": 0.1,
        "future_horizon": 5,
        "slice_encoder": {"type": "identity", "hidden_dim": 1024},
        "sequence_encoder": {
            "backend": "smoke",
            "layers": 2,
            "d_model": 512,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        },
        "state": {"d_anat": 256, "d_lesion": 192, "d_unc": 64, "world_dim": 512},
    }


def test_sliceworld_core_and_losses():
    torch.manual_seed(7)
    model = SliceWorldCore(smoke_config()).eval()
    features = torch.randn(2, 8, 1024)
    labels = torch.tensor([[0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.float32)
    lengths = torch.tensor([8, 7])
    outputs = model(slice_features=features, lengths=lengths)
    assert outputs["world_tokens"].shape == (2, 8, 512)
    assert outputs["lesion_zero_world_tokens"].shape == (2, 8, 512)
    assert outputs["pred_future_from_hidden"].shape == (2, 8, 5, 1024)
    assert outputs["lesion_presence"].min().item() >= 0.0
    assert outputs["lesion_presence"].max().item() <= 1.0
    lesion_zero_state = model.factor_heads.lesion_zero_state(
        model.factor_heads(outputs["hidden_states"])
    )
    expected_middle = torch.zeros_like(outputs["lesion"])
    assert torch.equal(lesion_zero_state[..., 256:448], expected_middle)
    expected_zero_tokens = model.world_projector(lesion_zero_state)
    assert torch.allclose(outputs["lesion_zero_world_tokens"], expected_zero_tokens, atol=1e-6)

    objective = SliceWorldObjective(
        loss_config_from_dict(
            {
                "alpha_nsp": 1,
                "alpha_fas": 1,
                "alpha_cf": 1,
                "alpha_ctrg": 0,
                "margin_delta": 0.1,
            }
        )
    )
    losses = objective(outputs, lesion_labels=labels, lengths=lengths)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["loss_cf_eff"].item() >= 0.0


def test_ctrg_prefix_projection_without_llm():
    config = {
        "world_model": smoke_config(),
        "llm": {"load": False, "hidden_dim": 2048},
    }
    model = SliceWorldForCTRG(config)
    features = torch.randn(1, 4, 1024)
    outputs = model.world_embeddings(slice_features=features, lengths=torch.tensor([4]))
    assert outputs["lm_prefix"].shape == (1, 4, 2048)


def test_prefix_outputs_ignore_future_slices():
    torch.manual_seed(7)
    model = SliceWorldCore(smoke_config()).eval()
    base = torch.randn(1, 7, 1024)
    changed = base.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10.0
    with torch.no_grad():
        base_outputs = model(slice_features=base, lengths=torch.tensor([7]))
        changed_outputs = model(slice_features=changed, lengths=torch.tensor([7]))
    for key in ["hidden_states", "world_tokens", "pred_future_from_hidden", "pred_future_from_world"]:
        assert torch.allclose(base_outputs[key][:, :4], changed_outputs[key][:, :4], atol=1e-6)

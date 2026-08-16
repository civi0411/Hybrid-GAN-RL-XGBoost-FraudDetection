"""
tests/test_models/test_gan.py
================================
Test kiến trúc GAN ở mức "smoke test": đảm bảo forward pass chạy được và
shape output đúng như spec — KHÔNG test hội tụ/chất lượng sinh dữ liệu
(việc đó cần train thật, nằm ngoài phạm vi unit test).
"""
import torch

from src.data.preprocessor import TabularDataSpec
from src.models.gan.discriminator import Discriminator
from src.models.gan.generator import Generator


def _dummy_spec() -> TabularDataSpec:
    return TabularDataSpec(
        numeric_columns=["amount", "hour"],
        categorical_columns=["card_type", "device"],
        categorical_cardinalities={"card_type": 4, "device": 3},
        target_column="isFraud",
    )


def test_generator_output_shapes():
    spec = _dummy_spec()
    gen = Generator(
        spec=spec, latent_dim=8, condition_dim=2, hidden_dims=[16, 16],
        use_batchnorm=False, gumbel_temperature=0.5,
    )
    batch_size = 5
    z = gen.sample_noise(batch_size, device="cpu")
    condition = torch.nn.functional.one_hot(torch.zeros(batch_size, dtype=torch.long), 2).float()

    numeric_out, categorical_outs = gen(z, condition, hard=False)

    assert numeric_out.shape == (batch_size, 2)
    assert len(categorical_outs) == 2
    assert categorical_outs[0].shape == (batch_size, 4)
    assert categorical_outs[1].shape == (batch_size, 3)
    # Gumbel-softmax output phải là phân phối xác suất hợp lệ (xấp xỉ tổng = 1)
    assert torch.allclose(categorical_outs[0].sum(dim=1), torch.ones(batch_size), atol=1e-4)


def test_generator_hard_gumbel_produces_onehot():
    spec = _dummy_spec()
    gen = Generator(spec=spec, latent_dim=8, condition_dim=2, hidden_dims=[16], use_batchnorm=False)
    batch_size = 4
    z = gen.sample_noise(batch_size, device="cpu")
    condition = torch.nn.functional.one_hot(torch.ones(batch_size, dtype=torch.long), 2).float()

    _, categorical_outs = gen(z, condition, hard=True)
    for cat_out in categorical_outs:
        # hard=True -> mỗi hàng phải là one-hot (đúng 1 giá trị = 1, còn lại = 0)
        assert torch.all((cat_out.sum(dim=1) - 1.0).abs() < 1e-4)
        assert torch.all((cat_out.max(dim=1).values - 1.0).abs() < 1e-4)


def test_discriminator_output_is_scalar_per_sample():
    spec = _dummy_spec()
    critic = Discriminator(spec=spec, condition_dim=2, hidden_dims=[16, 16])
    batch_size = 5
    numeric = torch.randn(batch_size, spec.numeric_dim)
    categorical = [torch.eye(4)[torch.randint(0, 4, (batch_size,))], torch.eye(3)[torch.randint(0, 3, (batch_size,))]]
    condition = torch.nn.functional.one_hot(torch.zeros(batch_size, dtype=torch.long), 2).float()

    scores = critic(numeric, categorical, condition)

    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()

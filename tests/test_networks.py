import numpy as np
import torch
from sapp.baselines.networks import NonlocalAttention, Encoder, MODELS


def test_nonlocal_matches_paper_matrix_expression():
    torch.manual_seed(8)
    m = NonlocalAttention().double()
    m.weight.data.fill_(0.4)
    x = torch.randn(2, 32, 2, 4, dtype=torch.float64)
    z = x.flatten(2)
    a = torch.softmax(m.q(z).transpose(1, 2) @ m.k(z), dim=1)
    exact = (z + 0.4 * m.out(m.v(z) @ a)).reshape_as(x)
    torch.testing.assert_close(m(x), exact, rtol=1e-12, atol=1e-12)


def test_pnn_feature_order_and_no_validation_refit():
    p = np.array([[[1.0, 7.0, 3.0, 0.0]], [[2.0, 6.0, 4.0, 0.0]]])
    e = Encoder("pnn", features=2).fit(p)
    before = dict(e.__dict__)
    image, power, delay = e.transform(p)
    np.testing.assert_allclose(image[:, :, 1:3], power)
    np.testing.assert_allclose(
        delay, np.broadcast_to(np.array([1, 2]), delay.shape) * 1 / e.dstd - e.dmean / e.dstd
    )
    e.transform(p * 10000)
    assert e.__dict__ == before


def test_all_architectures_have_finite_useful_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    p = np.random.default_rng(9).lognormal(0, 1, (4, 3, 28))
    for method in MODELS:
        enc = Encoder(method, features=9).fit(p)
        m = MODELS[method](3, 28, features=9)
        x = tuple(torch.from_numpy(a) for a in enc.transform(p))
        pred = m(x)
        assert pred.shape == (4, 2)
        pred.square().mean().backward()
        assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
        assert sum(float(t.grad.abs().sum()) for t in m.parameters() if t.grad is not None) > 0


def test_sst_sensor_permutation_property():
    torch.manual_seed(5)
    m = MODELS["sst"](6, 28).eval()
    x = torch.rand(3, 6, 28)
    torch.testing.assert_close(m((x,)), m((x[:, [3, 1, 0, 5, 4, 2]],)), rtol=2e-5, atol=1e-6)


def test_training_reduces_error_on_known_mapping():
    torch.set_num_threads(1)
    torch.manual_seed(6)
    p = np.random.default_rng(6).lognormal(0, 0.5, (12, 1, 28))
    y = torch.as_tensor(np.c_[p[:, 0, 2], p[:, 0, 8]], dtype=torch.float32)
    m = MODELS["pnn"](1, 28)
    x = tuple(torch.from_numpy(a) for a in Encoder("pnn").fit(p).transform(p))
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    initial = float((m(x) - y).square().mean())
    for _ in range(100):
        opt.zero_grad()
        loss = (m(x) - y).square().mean()
        loss.backward()
        opt.step()
    assert float((m(x) - y).square().mean()) < initial * 0.05


def test_minibatches_cover_each_sample_and_avoid_tiny_tails():
    from sapp.experiments.networks import minibatches

    for size in [3, 64, 126, 164, 256, 412, 515]:
        for batch in [64, 256]:
            order = torch.randperm(size)
            chunks = minibatches(order, batch)
            torch.testing.assert_close(torch.cat(chunks), order, rtol=0, atol=0)
            if len(chunks) > 1:
                assert min(map(len, chunks)) >= batch // 4

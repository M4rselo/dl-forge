import pytest
import torch
from dlfrg import optim as my


def make_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def train(model, opt, steps=100, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x, y = torch.randn(16, 4, generator=g), torch.randn(16, 1, generator=g)
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()


CASES = [
    # odkomentowuj w miarę postępów:
    # ("SGD", torch.optim.SGD, dict(lr=0.1)),                    # 12.4
    # ("SGD", torch.optim.SGD, dict(lr=0.1, momentum=0.9)),      # 12.6
    # ("Adam", torch.optim.Adam, dict(lr=1e-3)),                 # 12.10
]


@pytest.mark.parametrize("name, ref, kw", CASES)
def test_matches_pytorch(name, ref, kw):
    a, b = make_model(), make_model()
    train(a, getattr(my, name)(a.parameters(), **kw))
    train(b, ref(b.parameters(), **kw))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6), f"{name} {kw}: parameters differ"

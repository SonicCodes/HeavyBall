import pytest
import torch
from torch import nn
from torch._dynamo import config

import heavyball
import heavyball.utils
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch

config.cache_size_limit = 128


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(256, 2)])
def test_foreach(opt, size, depth: int, iterations: int = 128, outer_iterations: int = 3):
    set_torch()
    opt = getattr(heavyball, opt)

    peaks = []
    losses = []

    for do_ema in [True, False]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3)

            for _ in range(iterations):
                loss = model(torch.randn((1024, size), device='cuda')).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                if do_ema:
                    o.ema_update()
                    o.copy_emas_to_params()
                    o.copy_params_to_emas()
                losses[-1].append(loss.detach())

            if do_ema:
                o.copy_emas_to_params()
            loss = model(torch.randn((1024, size), device='cuda')).square().mean()
            losses[-1].append(loss.detach())

            del model, o
            clean()

    for i, (l0, l1) in enumerate(zip(*losses)):
        print(i, l0.item(), l1.item())
        assert l0.float() <= l1.float()
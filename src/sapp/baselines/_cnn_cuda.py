"""Capture the existing CNN/Adam update without changing its objective."""

import torch


class CapturedStep:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.x = torch.zeros(10, 1, 6, 12, device="cuda")
        self.y = torch.zeros(10, 2, device="cuda")
        self.mask = torch.ones(10, device="cuda")
        self.total = torch.zeros((), device="cuda")
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        states = {
            p: {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state.items()}
            for p, state in optimizer.state.items()
        }
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        model.load_state_dict(weights)
        for p, state in optimizer.state.items():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if p in states:
                        v.copy_(states[p][k])
                    else:
                        v.zero_()
        self.total.zero_()
        torch.cuda.synchronize()

    def _step(self):
        prediction = self.model(self.x)
        count = self.mask.sum()
        loss = (
            ((prediction - self.y).square().sum(dim=1) * self.mask).sum() / count + 1e-12
        ).sqrt()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.total.add_(loss.detach() * count)

    def epoch(self, x, y, order):
        shuffled_x = x[order.to("cuda")]
        shuffled_y = y[order.to("cuda")]
        self.total.zero_()
        for offset in range(0, len(x), 10):
            n = min(10, len(x) - offset)
            self.mask.fill_(1)
            if n == 10:
                self.x.copy_(shuffled_x[offset : offset + 10])
                self.y.copy_(shuffled_y[offset : offset + 10])
            else:
                self.mask[n:] = 0
                self.x.zero_()
                self.y.zero_()
                self.x[:n].copy_(shuffled_x[offset : offset + n])
                self.y[:n].copy_(shuffled_y[offset : offset + n])
            self.graph.replay()
        return float(self.total) / len(x)

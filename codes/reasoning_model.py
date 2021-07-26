import torch

from metrics import Metrics


class ReasoningModel(torch.nn.Module):
    def __init__(self):
        super(ReasoningModel, self).__init__()

    # self.E = 0

    def train(self, graph=None):
        pass

    def infer(self, infer_tris, graph=None):
        pass

    def enable_parameter(self, name):
        setattr(self, name, torch.nn.Parameter(getattr(self, name)))
        self._parameters['name'] = getattr(self, name)
        self.register_parameter(name, getattr(self, name))

    @property
    def metrics(self):
        if not hasattr(self, '_metrics') or self._metrics.N < self.E:
            self._metrics = Metrics(self.E)
        return self._metrics

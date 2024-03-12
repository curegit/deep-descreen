from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class AbsModule(Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def output_size(self, input_size: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def input_size(self, output_size: int) -> int:
        raise NotImplementedError()

    def reduced_padding(self, input_size: int) -> int:
        output_size = self.output_size(input_size)
        assert (input_size - output_size) % 2 == 0
        return (input_size - output_size) // 2

    def required_padding(self, output_size: int) -> int:
        input_size = self.input_size(output_size)
        assert (input_size - output_size) % 2 == 0
        return (input_size - output_size) // 2

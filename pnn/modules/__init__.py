from abc import ABC, abstractmethod
from torch.nn import Module


class BaseModel(ABC, Module):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()
    
    @abstractmethod
    def output_size(self, input_size):
        raise NotImplementedError()

    @abstractmethod
    def input_size(self, output_size):
        raise NotImplementedError()

    def reduced_padding(self, input_size):
        output_size = self.output_size(input_size)
        return (input_size - output_size) // 2

    def required_padding(self, output_size):
        input_size = self.input_size(output_size)
        return (input_size - output_size) // 2

from abc import ABC, abstractmethod
from torch.nn import Module

from ..utils import range_chunks


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


    def patch_slices(self, height, width, patch_size):
        for h_start, h_stop in range_chunks(height, patch_size):
            for w_start, w_stop in range_chunks(width, patch_size):
                h_pad = self.required_padding(h_stop - h_start)
                w_pad = self.required_padding(w_stop - w_start)
                h_slice = slice(h_start - h_pad, h_stop + h_pad)
                w_slice = slice(w_start - w_pad, w_stop + w_pad)
                yield h_slice, w_slice

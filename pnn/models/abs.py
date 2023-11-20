from ..utils import range_chunks
from ..modules import AbsModule


<<<<<<< HEAD
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

=======
class AbsModel(AbsModule):
>>>>>>> origin/desc

    def patch_slices(self, img_height, img_width, patch_size):
        for h_start, h_stop in range_chunks(height, patch_size):
            for w_start, w_stop in range_chunks(width, patch_size):
                h_pad = self.required_padding(h_stop - h_start)
                w_pad = self.required_padding(w_stop - w_start)
<<<<<<< HEAD
                h_s = slice(h_start, h_stop)
                w_s = slice(w_start, w_stop)
                h_slice = slice(h_start - h_pad + h_pad, h_stop + h_pad * 2)
                w_slice = slice(w_start - w_pad + h_pad, w_stop + w_pad * 2)
                yield (h_slice, w_slice), (h_s, w_s)
=======
                h_slice = slice(h_start - h_pad - h_pad, h_stop)
                w_slice = slice(w_start - w_pad - w_pad, w_stop)
                padded_input_slice = (h_slice, w_slice)
                output_slice = (slice(h_start, h_stop), slice(w_start, w_stop))
                yield padded_input_slice, output_slice
>>>>>>> origin/desc

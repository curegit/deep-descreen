from ..utils import range_chunks
from ..modules import AbsModule


class AbsModel(AbsModule):

    def patch_slices(self, img_height, img_width, patch_size):
        for h_start, h_stop in range_chunks(height, patch_size):
            for w_start, w_stop in range_chunks(width, patch_size):
                h_pad = self.required_padding(h_stop - h_start)
                w_pad = self.required_padding(w_stop - w_start)
                h_slice = slice(h_start - h_pad - h_pad, h_stop)
                w_slice = slice(w_start - w_pad - w_pad, w_stop)
                padded_input_slice = (h_slice, w_slice)
                output_slice = (slice(h_start, h_stop), slice(w_start, w_stop))
                yield padded_input_slice, output_slice

def input_size(output_size, kernel_size, stride=1, padding=0):
    return ((output_size - 1) * stride) + kernel_size - 2 * padding


def output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1

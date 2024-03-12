def input_size(output_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return ((output_size - 1) * stride) + kernel_size - 2 * padding


def output_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return (input_size - kernel_size + 2 * padding) // stride + 1

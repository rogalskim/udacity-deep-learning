import typing

import torch
import torch.nn as nn


def _calculate_2d_conv_output_dim_size(dim: int, input_size: (int, int), padding: (int, int), dilation: (int, int),
                                       kernel: (int, int), stride: (int, int)) -> int:
    return (input_size[dim] + 2 * padding[dim] - dilation[dim] * (kernel[dim] - 1) - 1) / stride[dim] + 1


def _calculate_2d_conv_output_shape(batch_size: int, output_channels: int, input_size: (int, int), padding: (int, int),
                                    dilation: (int, int), kernel: (int, int), stride: (int, int)) -> torch.Size:
    output_height = _calculate_2d_conv_output_dim_size(0, input_size, padding, dilation, kernel, stride)
    output_width = _calculate_2d_conv_output_dim_size(1, input_size, padding, dilation, kernel, stride)
    return torch.Size((batch_size, output_channels, int(output_height), int(output_width)))


def _calculate_2d_tconv_output_dim_size(dim: int, input_size: (int, int), padding: (int, int), dilation: (int, int),
                                        kernel: (int, int), stride: (int, int), output_padding: (int, int)) -> int:
    return (input_size[dim] - 1) * stride[dim] - 2 * padding[dim] + dilation[dim] * (kernel[dim] - 1) \
           + output_padding[dim] + 1


def _calculate_2d_tconv_output_shape(batch_size: int, output_channels: int, input_size: (int, int),
                                     padding: (int, int), dilation: (int, int), kernel: (int, int),
                                     stride: (int, int), output_padding: (int, int)) -> torch.Size:
    output_height = _calculate_2d_tconv_output_dim_size(0, input_size, padding, dilation, kernel, stride, output_padding)
    output_width = _calculate_2d_tconv_output_dim_size(1, input_size, padding, dilation, kernel, stride, output_padding)
    return torch.Size((batch_size, output_channels, int(output_height), int(output_width)))


def __make_tuple(source: typing.Union[int, float, tuple]) -> tuple:
    if type(source) in [int, float]:
        return source, source
    return source


def get_conv2d_output_shape(input_shape: torch.Size, layer: nn.Conv2d) -> torch.Size:
    assert len(input_shape) == 4, f"Input must have 4 dimensions, but has {len(input_shape)}"
    return _calculate_2d_conv_output_shape(batch_size=input_shape[0], output_channels=layer.out_channels,
                                           input_size=(input_shape[2], input_shape[3]),
                                           padding=layer.padding, dilation=layer.dilation,
                                           kernel=layer.kernel_size, stride=layer.stride)


def get_maxpool2d_output_shape(input_shape: torch.Size, layer: nn.MaxPool2d) -> torch.Size:
    assert len(input_shape) == 4, f"Input must have 4 dimensions, but has {len(input_shape)}"
    padding = __make_tuple(layer.padding)
    dilation = __make_tuple(layer.dilation)
    kernel = __make_tuple(layer.kernel_size)
    stride = __make_tuple(layer.stride)
    return _calculate_2d_conv_output_shape(batch_size=input_shape[0], output_channels=input_shape[1],
                                           input_size=(input_shape[2], input_shape[3]),
                                           padding=padding, dilation=dilation, kernel=kernel, stride=stride)


def get_conv_transpose2d_output_shape(input_shape: torch.Size, layer: nn.ConvTranspose2d) -> torch.Size:
    assert len(input_shape) == 4, f"Input must have 4 dimensions, but has {len(input_shape)}"
    return _calculate_2d_tconv_output_shape(batch_size=input_shape[0], output_channels=layer.out_channels,
                                            input_size=(input_shape[2], input_shape[3]),
                                            padding=layer.padding, dilation=layer.dilation,
                                            kernel=layer.kernel_size, stride=layer.stride,
                                            output_padding=layer.output_padding)


def get_upsample_output_shape(input_shape: torch.Size, layer: nn.Upsample) -> torch.Size:
    assert len(input_shape) == 4, f"Input Tensor must have 4 dimensions, but has {len(layer_input)}"
    batch_size = input_shape[0]
    channel_count = input_shape[1]
    input_height = input_shape[2]
    input_width = input_shape[3]

    scale = __make_tuple(layer.scale_factor)
    output_height = int(input_height * scale[0])
    output_width = int(input_width * scale[1])
    return torch.Size((batch_size, channel_count, output_height, output_width))


def __ignore_module(input_shape: torch.Size, _module: nn.Module) -> torch.Size:
    return input_shape


layer_type_to_function_map = {nn.Conv2d.__name__: get_conv2d_output_shape,
                              nn.MaxPool2d.__name__: get_maxpool2d_output_shape,
                              nn.ConvTranspose2d.__name__: get_conv_transpose2d_output_shape,
                              nn.Upsample.__name__: get_upsample_output_shape,
                              nn.Dropout.__name__: __ignore_module,
                              nn.BatchNorm2d.__name__: __ignore_module,
                              nn.ModuleList.__name__: __ignore_module,
                              nn.ReLU.__name__: __ignore_module,
                              nn.LeakyReLU.__name__: __ignore_module,
                              nn.Tanh.__name__: __ignore_module}


def get_layer_output_shapes(data_shape: torch.Size, network: nn.Module) -> dict:
    def handle_nested_layer_list(input_shape: torch.Size, layer_list: nn.ModuleList) -> (torch.Size, dict):
        sublayer_shape_dict = get_layer_output_shapes(input_shape, layer_list)
        sublayers = list(layer.named_children())
        last_sublayer_name = sublayers[-1][0]
        output_shape = sublayer_shape_dict[last_sublayer_name]
        return output_shape, sublayer_shape_dict

    layers = list(network.named_children())
    shape_dict = {}
    for layer_name, layer in layers:
        layer_type_name = layer.__class__.__name__
        if layer_type_name == nn.ModuleList.__name__:
            if len(layer) == 0:
                continue
            list_output_shape, list_shape_dict = handle_nested_layer_list(data_shape, layer)
            data_shape = list_output_shape
            shape_dict[layer_name] = list_shape_dict
        else:
            shape_function = layer_type_to_function_map[layer_type_name]
            data_shape = shape_function(data_shape, layer)
            shape_dict[layer_name] = data_shape

    return shape_dict


module_list_callable = get_layer_output_shapes

#!/usr/bin/env python3
import argparse
import gdown
import os
import tfvan
import torch

CHECKPOINTS = {
    'van_tiny': 'https://drive.google.com/file/d/1KYoIe1Zl3ZaPCwRuvnpkLyOEK04JKemu/view?usp=sharing',
    'van_small': 'https://drive.google.com/file/d/1LFsJHwxAs1TcXAjJ28G86_jwYwV8DzuG/view?usp=sharing',
    'van_base': 'https://drive.google.com/file/d/1qApsgXCbngNYOji2UzJsfeEsPOu6dBo3/view?usp=sharing',
    'van_large': 'https://drive.google.com/file/d/10n6u-W3IrqiCD-7wkotejV_1XiS9kuWF/view?usp=sharing',
}
TF_MODELS = {
    'van_tiny': tfvan.VanTiny,
    'van_small': tfvan.VanSmall,
    'van_base': tfvan.VanBase,
    'van_large': tfvan.VanLarge
}


def convert_name(name):
    name = name.replace(':0', '').replace('/', '.')
    name = name.replace('depthwise_kernel', 'weight').replace('kernel', 'weight')
    name = name.replace('moving_mean', 'running_mean').replace('moving_variance', 'running_var')
    name = name.replace('gamma', 'weight').replace('beta', 'bias')

    return name


def convert_weight(weight, name):
    if '.layer_scale' in name and 1 == len(weight.shape):
        return weight[None, None, None]

    if '.dwconv.weight' in name or '.spatial_gating_unit.conv0.weight' in name \
            or '.spatial_gating_unit.conv_spatial.weight' in name:
        return weight.transpose([2, 3, 0, 1])

    if '.weight' in name and 4 == len(weight.shape):
        return weight.transpose([2, 3, 1, 0])

    if '.weight' in name and 2 == len(weight.shape):
        return weight.T

    return weight


if '__main__' == __name__:
    parser = argparse.ArgumentParser(
        description='Visual Attention Network weight conversion from PyTorch to TensorFlow')
    parser.add_argument(
        'model_type',
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help='Model checkpoint to load')
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save TensorFlow model weights')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(argv.out_path), 'Wrong output path'

    weights_path = os.path.join(argv.out_path, f'{argv.model_type}.pth.tar')
    gdown.download(url=CHECKPOINTS[argv.model_type], output=weights_path, quiet=False, fuzzy=True, resume=True)
    weights_torch = torch.load(weights_path, map_location=torch.device('cpu'))

    model = TF_MODELS[argv.model_type](weights=None)

    weights_tf = []
    for w in model.weights:
        name = convert_name(w.name)
        assert name in weights_torch['state_dict'], f'Can\'t find weight {name} in checkpoint'

        weight = weights_torch['state_dict'].pop(name).numpy()
        weight = convert_weight(weight, name)

        weights_tf.append(weight)

    model.set_weights(weights_tf)
    model.save_weights(weights_path.replace('.pth.tar', '.h5'), save_format='h5')

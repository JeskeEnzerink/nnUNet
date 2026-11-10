#!/usr/bin/env python3
"""Single-grid, out-of-core inference for one-channel 3D Zarr images.

Uses the model patch size, global normalization, batched network calls and
Gaussian-weighted logits. Supports DefaultPreprocessor at matching spacing;
resampling, masked Z-score, cascades and custom preprocessors fail explicitly.
Output '0' is a final segmentation in the input's z,y,x coordinates. Intermediate
arrays remain in the new output store for inspection; originals are not changed.
"""

import itertools
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from time import perf_counter

import numpy as np


def tiles(shape, size=(128, 128, 128), origin=(0, 0, 0)):
    """Non-overlapping I/O tiles, independent of the prediction grid."""
    for start in itertools.product(*(range(0, n, b) for n, b in zip(shape, size))):
        yield tuple(slice(o + s, o + min(s + b, n))
                    for o, s, b, n in zip(origin, start, size, shape))


def read_scaled(source, sl, bounds):
    data = np.array(source[sl], dtype=np.float32, copy=True)
    if not np.isfinite(data).all():
        raise ValueError('Input contains NaN or infinite intensities')
    if bounds is not None:
        # Preserve the original script's external intensity transform.
        lower, upper = bounds
        np.clip(data, lower, upper, out=data)
        data = (data - lower) / upper
    return data


def validate_model(predictor, source, input_spacing, bounds):
    config, plans = predictor.configuration_manager, predictor.plans_manager
    print('Model patch size:', config.patch_size, flush=True)
    print('Model spacing (after axis transpose):', config.spacing, flush=True)
    print('Axis transpose:', plans.transpose_forward, flush=True)
    print('Normalization:', config.normalization_schemes,
          'masked:', config.use_mask_for_norm, flush=True)
    if source.ndim != 3 or len(config.patch_size) != 3 or any(n < 1 for n in source.shape):
        raise ValueError('This script requires a nonempty 3D Zarr array and a 3D model')
    if len(predictor.dataset_json['channel_names']) != 1 or config.previous_stage_name is not None:
        raise ValueError('Only single-channel, non-cascaded models are supported')
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    if config.preprocessor_class is not DefaultPreprocessor:
        raise ValueError('Custom preprocessing needs a matching streaming implementation')
    scheme = config.normalization_schemes[0]
    supported = {'ZScoreNormalization', 'CTNormalization', 'NoNormalization',
                 'RescaleTo01Normalization', 'RGBTo01Normalization'}
    if scheme not in supported or (scheme == 'ZScoreNormalization' and config.use_mask_for_norm[0]):
        raise ValueError('Unsupported normalization: global masked/custom normalization is not approximated patchwise')
    if input_spacing is None:
        raise ValueError('Set input_spacing to verified z,y,x voxel spacing in the SAME units as the training data; do not guess from model spacing')
    if len(input_spacing) != 3 or not np.isfinite(input_spacing).all() or min(input_spacing) <= 0:
        raise ValueError('input_spacing must contain three positive finite values')
    spacing = np.asarray(input_spacing)[list(plans.transpose_forward)]
    target = [s if t is None else t for s, t in zip(spacing, config.spacing)]
    if len(target) != 3 or not np.allclose(spacing, target, rtol=1e-6, atol=0):
        raise ValueError(f'Input spacing in model order {spacing} differs from {target}. '
                         'A separately validated whole-volume resampling stage is required; prediction has not started')
    if bounds is not None:
        lower, upper = bounds
        if not np.isfinite(bounds).all() or upper <= 0 or lower > upper:
            raise ValueError('Invalid external intensity bounds')


def find_crop(source, bounds):
    """Find the global nonzero bounding box after the external transform."""
    lo, hi = np.array(source.shape), np.zeros(3, dtype=int)
    for sl in tiles(source.shape):
        data = read_scaled(source, sl, bounds)
        mask = data != 0
        if mask.any():
            for axis in range(3):
                occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
                lo[axis] = min(lo[axis], sl[axis].start + int(occupied[0]))
                hi[axis] = max(hi[axis], sl[axis].start + int(occupied[-1]) + 1)
    return None if np.any(hi <= lo) else (lo, hi)


def global_statistics(source, bounds, crop):
    """Combine moments without retaining the whole cropped image in RAM."""
    lo, hi = crop
    count, mean, m2 = 0, 0.0, 0.0
    minimum, maximum = float('inf'), float('-inf')
    for sl in tiles(hi - lo, origin=lo):
        data = read_scaled(source, sl, bounds).astype(np.float64)
        n, local_mean = data.size, float(data.mean())
        delta = local_mean - mean
        total = count + n
        m2 += float(np.square(data - local_mean).sum()) + delta * delta * count * n / total
        mean += delta * n / total
        count = total
        minimum, maximum = min(minimum, float(data.min())), max(maximum, float(data.max()))
    return dict(mean=mean, std=math.sqrt(m2 / count), minimum=minimum, maximum=maximum)


def normalize(data, scheme, stats, intensity_properties):
    if scheme == 'ZScoreNormalization':
        data -= stats['mean']
        data /= max(stats['std'], 1e-8)
    elif scheme == 'RescaleTo01Normalization':
        data -= stats['minimum']
        data /= max(stats['maximum'] - stats['minimum'], 1e-8)
    elif scheme == 'CTNormalization':
        p = intensity_properties
        np.clip(data, p['percentile_00_5'], p['percentile_99_5'], out=data)
        data -= p['mean']
        data /= max(p['std'], 1e-8)
    elif scheme == 'RGBTo01Normalization':
        if data.min() < 0 or data.max() > 255:
            raise ValueError('RGBTo01Normalization expects intensities in [0, 255]')
        data /= 255
    elif scheme != 'NoNormalization':
        raise ValueError(f'Unsupported normalization: {scheme}')
    return data


def prepare_volume(source, destination, bounds, crop, axes, scheme, stats, intensity_properties):
    lo, hi = crop
    for sl in tiles(hi - lo, origin=lo):
        data = normalize(read_scaled(source, sl, bounds), scheme, stats, intensity_properties)
        relative = tuple(slice(s.start - int(o), s.stop - int(o)) for s, o in zip(sl, lo))
        destination[tuple(relative[i] for i in axes)] = data.transpose(axes)


def patch_grid(shape, patch_size, tile_step_size):
    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
    padded = tuple(max(n, p) for n, p in zip(shape, patch_size))
    before = tuple((p - n) // 2 for n, p in zip(shape, padded))
    steps = compute_steps_for_sliding_window(padded, patch_size, tile_step_size)
    return steps, before


def read_patch_batch(source, starts, patch_size, before):
    batch = np.zeros((len(starts), 1, *patch_size), dtype=np.float32)
    destinations = []
    for index, start in enumerate(starts):
        # Intersect a patch in the virtual padded image with the actual crop.
        first = tuple(s - b for s, b in zip(start, before))
        lo = tuple(max(0, s) for s in first)
        hi = tuple(min(n, s + p) for n, s, p in zip(source.shape, first, patch_size))
        target = tuple(slice(a, b) for a, b in zip(lo, hi))
        local = tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, first))
        batch[(index, 0, *local)] = source[target]
        destinations.append((target, local))
    return batch, destinations


def prefetched_batches(source, steps, patch_size, before, batch_size):
    starts = itertools.product(*steps)
    with ThreadPoolExecutor(max_workers=1) as pool:
        group = list(itertools.islice(starts, batch_size))
        future = pool.submit(read_patch_batch, source, group, patch_size, before) if group else None
        while future is not None:
            result = future.result()
            group = list(itertools.islice(starts, batch_size))
            future = pool.submit(read_patch_batch, source, group, patch_size, before) if group else None
            yield result


def accumulate(logit_sum, weight_sum, predictions, destinations, gaussian):
    # Sequential writes are required: neighboring patches share chunks.
    for prediction, (target, local) in zip(predictions, destinations):
        weight = gaussian[local]
        logit_sum[(slice(None), *target)] += prediction[(slice(None), *local)] * weight
        weight_sum[target] += weight


def infer(predictor, source, logit_sum, weight_sum, batch_size):
    import torch
    from tqdm import tqdm
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import is_optimized_module
    patch = tuple(predictor.configuration_manager.patch_size)
    steps, before = patch_grid(source.shape, patch, predictor.tile_step_size)
    total = math.prod(len(s) for s in steps)
    print(f'One prediction grid: {total} patches, patch size {patch}, GPU batch size {batch_size}', flush=True)
    gaussian = compute_gaussian(patch, sigma_scale=1. / 8, value_scaling_factor=10,
                                dtype=torch.float32, device=torch.device('cpu')).numpy()
    predictor.network.to(predictor.device).eval()
    parameters = predictor.list_of_parameters
    if not parameters:
        raise ValueError('No trained model parameters were loaded')
    network = predictor.network._orig_mod if is_optimized_module(predictor.network) else predictor.network
    if len(parameters) == 1:
        network.load_state_dict(parameters[0])
    batches = prefetched_batches(source, steps, patch, before, batch_size)
    try:
        with torch.inference_mode(), tqdm(total=total, desc='Predicting model patches') as progress:
            for data, destinations in batches:
                tensor = torch.from_numpy(data).to(predictor.device)
                prediction = None
                context = torch.autocast('cuda') if predictor.device.type == 'cuda' else nullcontext()
                with context:
                    for params in parameters:
                        if len(parameters) > 1:
                            network.load_state_dict(params)
                        # Direct network prediction: no nested sliding window,
                        # no per-patch crop, resampling or normalization.
                        logits = predictor._internal_maybe_mirror_and_predict(tensor).float()
                        if prediction is None:
                            prediction = logits
                        else:
                            prediction.add_(logits)
                    prediction.div_(len(parameters))
                predictions = prediction.cpu().numpy()
                if not np.isfinite(predictions).all():
                    raise RuntimeError('Model produced non-finite logits')
                accumulate(logit_sum, weight_sum, predictions, destinations, gaussian)
                progress.update(len(destinations))
                del data, tensor, logits, prediction, predictions
    finally:
        batches.close()


def finalize(logit_sum, weight_sum, output, crop, axes, label_manager):
    inverse = tuple(np.argsort(axes))
    lo, _ = crop
    for sl in tiles(weight_sum.shape):
        weights = weight_sum[sl]
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise RuntimeError('Prediction grid left invalid or uncovered voxels')
        logits = logit_sum[(slice(None), *sl)] / weights[None]
        labels = label_manager.convert_logits_to_segmentation(logits)
        target = tuple(slice(sl[inverse[i]].start + int(lo[i]), sl[inverse[i]].stop + int(lo[i]))
                       for i in range(3))
        output[target] = labels.transpose(inverse)


def run(predictor, input_location, output_location, input_spacing, bounds=None, gpu_batch_size=4):
    import zarr
    if isinstance(gpu_batch_size, bool) or not isinstance(gpu_batch_size, int) or gpu_batch_size < 1:
        raise ValueError('gpu_batch_size must be a positive integer')
    input_group = zarr.open_group(input_location, mode='r')
    source = input_group['0']
    validate_model(predictor, source, input_spacing, bounds)
    config, plans = predictor.configuration_manager, predictor.plans_manager
    axes = tuple(plans.transpose_forward)
    print('Finding global crop...', flush=True)
    crop = find_crop(source, bounds)
    # Exclusive creation: never overwrite an existing store or an input image.
    group = zarr.open_group(output_location, mode='w-', zarr_format=3)
    group.attrs.update({'complete': False, 'input': os.path.abspath(input_location),
                        'spacing_zyx': list(input_spacing), 'model_patch_size': list(config.patch_size),
                        'transpose_forward': list(axes), 'external_intensity_bounds': bounds})

    def create(name, shape, chunks, dtype, dimensions):
        return group.create_array(name=name, shape=tuple(int(i) for i in shape),
                                  chunks=tuple(int(i) for i in chunks), dtype=dtype,
                                  dimension_names=dimensions, fill_value=0,
                                  compressors=[zarr.codecs.BloscCodec(cname='lz4', clevel=5)])

    output = create('0', source.shape, tuple(min(n, 128) for n in source.shape), np.int32, ('z', 'y', 'x'))
    if crop is None:
        group.attrs['complete'] = True
        print('No nonzero input after intensity transform; wrote background segmentation', flush=True)
        return
    lo, hi = crop
    shape = tuple(int((hi - lo)[i]) for i in axes)
    chunks = tuple(min(n, p, 128) for n, p in zip(shape, config.patch_size))
    heads = predictor.label_manager.num_segmentation_heads
    estimated = 4 * (math.prod(shape) * (heads + 2) + math.prod(source.shape))
    print(f'Intermediate arrays plus segmentation: approximately {estimated / 2**30:.1f} GiB uncompressed', flush=True)
    group.attrs['crop_zyx'] = [lo.tolist(), hi.tolist()]
    dimensions = tuple(('z', 'y', 'x')[i] for i in axes)
    normalized = create('preprocessed', shape, chunks, np.float32, dimensions)
    scheme = config.normalization_schemes[0]
    stats = global_statistics(source, bounds, crop) if scheme in {'ZScoreNormalization', 'RescaleTo01Normalization'} else {}
    group.attrs['normalization_statistics'] = stats
    properties = (plans.foreground_intensity_properties_per_channel or {}).get('0', {})
    print('Writing globally normalized volume once...', flush=True)
    prepare_volume(source, normalized, bounds, crop, axes, scheme, stats, properties)
    logit_sum = create('logit_sum', (heads, *shape), (1, *chunks), np.float32, ('class', *dimensions))
    weight_sum = create('weight_sum', shape, chunks, np.float32, dimensions)
    infer(predictor, normalized, logit_sum, weight_sum, gpu_batch_size)
    print('Converting weighted logits to final labels...', flush=True)
    finalize(logit_sum, weight_sum, output, crop, axes, predictor.label_manager)
    group.attrs['complete'] = True
    print(f'Finished: {output_location} (array 0 is the final segmentation)', flush=True)


def main():
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    input_location = '/home/wirrbel/2026-11-02_jeske_stz_test_P0005/I10_P0005_C00_WFA-488.ome.zarr/'
    results_dir = '/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_results/Dataset020_wfa_moritz_ilastik/nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__3d_fullres/'
    input_spacing = None  # REQUIRED: verified z,y,x spacing, in the training data's units.
    gpu_batch_size = 4
    input_location = os.path.abspath(input_location)
    with open(input_location + '.intensity.json', encoding='utf-8') as stream:
        intensity = json.load(stream)
    bounds = (intensity['local_min'], intensity['local_95'])
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=False, device=torch.device('cuda', 0),
        verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(results_dir, use_folds=(0,), checkpoint_name='checkpoint_best.pth')
    # Validate before creating output directories or scanning the large image.
    import zarr
    input_group = zarr.open_group(input_location, mode='r')
    attributes = dict(input_group.attrs)
    metadata = attributes.get('ome', attributes).get('multiscales', [])
    for entry in metadata:
        for dataset in entry.get('datasets', []):
            if dataset.get('path') == '0':
                print('Input OME axes/units:', entry.get('axes'), flush=True)
                print('Input OME transforms:', entry.get('coordinateTransformations', []),
                      dataset.get('coordinateTransformations', []), flush=True)
    validate_model(predictor, input_group['0'], input_spacing, bounds)
    directory = tempfile.mkdtemp(prefix=os.path.basename(input_location) + '.single-grid-',
                                 dir=os.path.dirname(input_location))
    output_location = os.path.join(directory, 'prediction.zarr')
    print('New output:', output_location, flush=True)
    start = perf_counter()
    run(predictor, input_location, output_location, input_spacing, bounds, gpu_batch_size)
    print(f'Total time: {perf_counter() - start:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()

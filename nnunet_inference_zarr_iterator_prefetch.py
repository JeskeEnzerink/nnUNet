#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-batched copy of wirrbel's Zarr inference script.

CPU preprocessing can run ahead in spawned processes; GPU inference and Zarr
writes stay in the parent. Sliding-window patches within each Zarr block are
batched on the GPU. The original normalization, label sums and empty-block sentinel
are preserved (these outputs are not probability maps).
"""

import itertools
import json
import multiprocessing
import os
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import torch
import zarr
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from tqdm import tqdm


class BatchedNNUNetPredictor(nnUNetPredictor):
    """Batch patches within one block, preserving nnUNet's weighted stitching.

    Uses the installed predictor for preprocessing, padding, mirroring, folds,
    autocast and export. This override targets the repository's current private
    sliding-window method; recheck compatibility after updating nnUNet.
    """

    def __init__(self, *args, gpu_batch_size=4, **kwargs):
        if not isinstance(gpu_batch_size, int) or isinstance(gpu_batch_size, bool) or gpu_batch_size < 1:
            raise ValueError('gpu_batch_size must be a positive integer')
        super().__init__(*args, **kwargs)
        self.gpu_batch_size = gpu_batch_size

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device=True):
        if self.gpu_batch_size == 1:
            return super()._internal_predict_sliding_window_return_logits(data, slicers, do_on_device)

        results_device = self.device if do_on_device else torch.device('cpu')
        predicted_logits = n_predictions = gaussian = workon = predictions = prediction = None
        try:
            empty_cache(self.device)
            data = data.to(results_device)
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half, device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            gaussian = compute_gaussian(
                tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                value_scaling_factor=10, device=results_device
            ) if self.use_gaussian else 1

            with tqdm(total=len(slicers), disable=not self.allow_tqdm) as progress:
                for start in range(0, len(slicers), self.gpu_batch_size):
                    batch_slicers = slicers[start:start + self.gpu_batch_size]
                    # All sliding-window patches have the model's patch size,
                    # including border patches after nnUNet's padding step.
                    workon = torch.stack([data[sl] for sl in batch_slicers]).to(self.device)
                    predictions = self._internal_maybe_mirror_and_predict(workon).to(results_device)
                    for prediction, sl in zip(predictions, batch_slicers):
                        if self.use_gaussian:
                            prediction *= gaussian
                        predicted_logits[sl] += prediction
                        n_predictions[sl[1:]] += gaussian
                    # Release this batch before allocating the next one.
                    workon = predictions = prediction = None
                    progress.update(len(batch_slicers))

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if not torch.isfinite(predicted_logits).all():
                raise RuntimeError('Non-finite sliding-window logits; check Gaussian weights and precision')
            return predicted_logits
        except Exception:
            del data, predicted_logits, n_predictions, gaussian, workon, predictions, prediction
            empty_cache(self.device)
            empty_cache(results_device)
            raise


def generate_slice_blocks(shape, block_size=(200, 200, 200), overlap=0.5):
    # Preserve the original convention: overlap is the step fraction.
    steps = [int(b * overlap) for b in block_size]
    if len(shape) != 3 or len(block_size) != 3 or any(s <= 0 for s in steps):
        raise ValueError('Expected 3D blocks with positive steps')
    if not 0 < overlap <= 1:
        raise ValueError('overlap must be in (0, 1]')
    ranges = [range(0, s, step) for s, step in zip(shape, steps)]
    for start in itertools.product(*ranges):
        yield tuple(slice(s, s + b) for s, b in zip(start, block_size))


def create_zarr_group(output_location, input_data, output_dtype):
    group = zarr.open_group(output_location, mode='w-', zarr_format=3)
    return group.create_array(
        name='0', shape=input_data.shape, chunks=input_data.chunks,
        shards=input_data.shards, dtype=output_dtype,
        dimension_names=('z', 'y', 'x'),
        compressors=[zarr.codecs.BloscCodec(cname='lz4', clevel=5)]
    )


class _ZarrPreprocessor:
    def __init__(self, input_location, lower_bound, upper_bound, spacing,
                 plans_manager, configuration_manager, dataset_json, verbose):
        self.input_data = zarr.open_group(input_location, mode='r')['0']
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.spacing = spacing
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.preprocessor = configuration_manager.preprocessor_class(verbose=verbose)

    def __call__(self, block):
        image = self.input_data[block]
        if image.max() == 0:
            return block, None
        image = image.astype(np.float32)
        np.clip(image, self.lower_bound, self.upper_bound, out=image)
        image = (image - self.lower_bound) / self.upper_bound
        # First axis is the channel axis, not a batch axis.
        data, _, properties = self.preprocessor.run_case_npy(
            image[None], None, {'spacing': list(self.spacing)},
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        # Transfer NumPy arrays between processes; construct tensors in parent.
        return block, {'data': data, 'data_properties': properties, 'ofile': None}


def _init_preprocessing_worker(args):
    global _worker_preprocessor
    torch.set_num_threads(1)
    _worker_preprocessor = _ZarrPreprocessor(*args)


def _preprocess_block(block):
    return _worker_preprocessor(block)


def zarr_iterator(blocks, preprocessing_args, num_processes=2):
    """Yield (block, preprocessed item), with bounded, ordered read-ahead."""
    if num_processes < 0:
        raise ValueError('num_processes must be nonnegative')
    if num_processes == 0:
        preprocess = _ZarrPreprocessor(*preprocessing_args)
        for block in blocks:
            yield preprocess(block)
        return

    blocks = iter(blocks)
    with ProcessPoolExecutor(
        max_workers=num_processes, mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_preprocessing_worker, initargs=(preprocessing_args,)
    ) as pool:
        pending = deque()
        try:
            for block in itertools.islice(blocks, 2 * num_processes):
                pending.append(pool.submit(_preprocess_block, block))
            while pending:
                result = pending.popleft().result()
                block = next(blocks, None)
                if block is not None:
                    pending.append(pool.submit(_preprocess_block, block))
                yield result
        finally:
            for future in pending:
                future.cancel()


def predict_zarr(predictor, input_location, output_values, output_counts,
                 lower_bound, upper_bound, spacing=(1.0, 1.0, 1.0),
                 block_size=(200, 200, 200), overlap=0.5,
                 blocks_per_call=8, num_processes_preprocessing=2,
                 num_processes_segmentation_export=2):
    """Accumulate the same segmentation-label sums as the original script."""
    if blocks_per_call < 1 or num_processes_segmentation_export < 1:
        raise ValueError('blocks_per_call and export process count must be positive')
    if num_processes_preprocessing < 0:
        raise ValueError('preprocessing process count must be nonnegative')
    if not np.isfinite([lower_bound, upper_bound]).all() or upper_bound <= 0 or lower_bound > upper_bound:
        raise ValueError('Expected finite intensity bounds, lower <= upper and upper > 0')
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('spacing must contain three positive finite values in z, y, x order')
    if predictor.configuration_manager.previous_stage_name is not None:
        raise ValueError('Cascaded models require a previous-stage segmentation')

    args = (input_location, lower_bound, upper_bound, spacing,
            predictor.plans_manager, predictor.configuration_manager,
            predictor.dataset_json, predictor.verbose_preprocessing)
    blocks = generate_slice_blocks(output_values.shape, block_size, overlap)
    prepared = zarr_iterator(blocks, args, num_processes_preprocessing)
    try:
        while True:
            # nnUNet returns a list only when this entire iterator is exhausted.
            # Limit it to a small group so results do not grow with volume size.
            group = list(itertools.islice(prepared, blocks_per_call))
            if not group:
                break
            active = [(block, item) for block, item in group if item is not None]

            def data_iterator():
                for _, item in active:
                    data = torch.from_numpy(item['data']).contiguous()
                    if predictor.device.type == 'cuda':
                        data = data.pin_memory()
                    yield {**item, 'data': data}

            predictions = predictor.predict_from_data_iterator(
                data_iterator(), save_probabilities=False,
                num_processes_segmentation_export=num_processes_segmentation_export
            ) if active else []
            if len(predictions) != len(active):
                raise RuntimeError('nnUNet returned an unexpected number of predictions')
            predictions = iter(predictions)
            # Only this process writes, in original block order. Overlapping
            # chunk/shard read-modify-write operations must not race.
            for block, item in group:
                segmentation = -1000.0 if item is None else next(predictions)
                output_values[block] += segmentation
                output_counts[block] += 1
            del group, active, predictions
    finally:
        prepared.close()


def load_json(json_location):
    with open(json_location, encoding='utf-8') as stream:
        return json.load(stream)


def main():
    input_zarr_location = '/home/wirrbel/2026-11-02_jeske_stz_test_P0005/I10_P0005_C00_WFA-488.ome.zarr/'
    results_dir = '/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_results/Dataset020_wfa_moritz_ilastik/nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__3d_fullres/'
    spacing = (1.0, 1.0, 1.0)  # Replace with actual voxel spacing in z, y, x order.
    num_processes_preprocessing = 2  # Set to 0 to compare serial preprocessing.
    blocks_per_call = 8  # Bounds retained results; this is not a GPU batch size.
    gpu_batch_size = 4  # Actual patches per GPU call; try 2 if GPU memory runs out.

    input_zarr_location = os.path.abspath(input_zarr_location)
    output_zarr_location, dataset_name = os.path.split(input_zarr_location)
    input_data = zarr.open_group(input_zarr_location, mode='r')['0']
    bounds = load_json(input_zarr_location + '.intensity.json')

    predictor = BatchedNNUNetPredictor(
        gpu_batch_size=gpu_batch_size,
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device('cuda', 0),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        results_dir, use_folds=(0,), checkpoint_name='checkpoint_best.pth'
    )

    values_location = tempfile.mkdtemp(prefix=dataset_name + '.inference_values-', dir=output_zarr_location)
    counts_location = tempfile.mkdtemp(prefix=dataset_name + '.count_map-', dir=output_zarr_location)
    output_values = create_zarr_group(values_location, input_data, np.float16)
    output_counts = create_zarr_group(counts_location, input_data, np.uint8)
    print(f'Inference values: {values_location}\nCount map: {counts_location}')
    start = perf_counter()
    predict_zarr(
        predictor, input_zarr_location, output_values, output_counts,
        bounds['local_min'], bounds['local_95'], spacing=spacing,
        blocks_per_call=blocks_per_call,
        num_processes_preprocessing=num_processes_preprocessing
    )
    print(f'Inference completed in {perf_counter() - start:.1f} seconds')


if __name__ == '__main__':
    main()

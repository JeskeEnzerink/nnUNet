#!/usr/bin/env python3
"""Quick real-model smoke test of nnunet_inference_zarr_single_grid.infer.

Keep both scripts together. Uses synthetic, already-preprocessed data in RAM;
does not open a Zarr image or write prediction files. Model loading time varies.
"""

import argparse
import math
import os
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', help='Trained model folder containing plans.json and fold_0/')
    parser.add_argument('--checkpoint', default='checkpoint_best.pth')
    parser.add_argument('--fold', default='0', help='One fold number, or all')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--mirroring', action='store_true', help='Also test production mirroring; slower')
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error('--batch-size must be at least 2 to demonstrate batching')
    if args.fold != 'all' and not args.fold.isdigit():
        parser.error('--fold must be a fold number or all')

    # Avoid a long compilation warm-up just for the smoke test.
    os.environ['nnUNet_compile'] = 'false'
    start = perf_counter()
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunet_inference_zarr_single_grid import infer, patch_grid

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment; no GPU test was run')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    print('GPU:', torch.cuda.get_device_name(device), flush=True)
    print('Loading trained model (compilation disabled for this short test)...', flush=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=args.mirroring,
        perform_everything_on_device=False, device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    fold = 'all' if args.fold == 'all' else int(args.fold)
    predictor.initialize_from_trained_model_folder(
        args.model_dir, use_folds=(fold,), checkpoint_name=args.checkpoint)
    config = predictor.configuration_manager
    patch = tuple(int(n) for n in config.patch_size)
    if len(patch) != 3 or min(patch) < 2:
        raise ValueError('This smoke test targets the single-grid script for 3D models')
    if len(predictor.dataset_json['channel_names']) != 1 or config.previous_stage_name is not None:
        raise ValueError('The single-grid script supports one-channel, non-cascaded models only')

    # At step_size=0.5 this makes batch_size+1 patches: one full batch
    # and one partial batch, including overlapping logit accumulation.
    shape = (patch[0] + int(args.batch_size * patch[0] * 0.5), *patch[1:])
    steps, _ = patch_grid(shape, patch, predictor.tile_step_size)
    count = math.prod(len(axis) for axis in steps)
    expected_batches = [min(args.batch_size, count - i) for i in range(0, count, args.batch_size)]
    if expected_batches != [args.batch_size, 1]:
        raise RuntimeError(f'Unexpected smoke-test grid: {expected_batches}')
    source = np.random.default_rng(42).standard_normal(shape, dtype=np.float32)
    heads = predictor.label_manager.num_segmentation_heads
    sums = np.zeros((heads, *shape), dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)
    observed = []

    def check_forward(network, inputs, output):
        data = inputs[0]
        if data.device != device or not isinstance(output, torch.Tensor) or output.device != device:
            raise RuntimeError('A network input or output was not on the requested CUDA device')
        if tuple(data.shape[1:]) != (1, *patch):
            raise RuntimeError(f'Unexpected network input shape: {tuple(data.shape)}')
        if tuple(output.shape) != (data.shape[0], heads, *patch):
            raise RuntimeError(f'Unexpected network output shape: {tuple(output.shape)}')
        observed.append(int(data.shape[0]))

    handle = predictor.network.register_forward_hook(check_forward)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    inference_start = perf_counter()
    try:
        # This is the production single-grid function, not a substitute loop.
        infer(predictor, source, sums, weights, args.batch_size)
        torch.cuda.synchronize(device)
    finally:
        handle.remove()
    elapsed = perf_counter() - inference_start
    mirrors = predictor.allowed_mirroring_axes
    repeats = 2 ** len(mirrors) if args.mirroring and mirrors is not None else 1
    expected = [size for size in expected_batches for _ in range(repeats)]
    if observed != expected:
        raise RuntimeError(f'GPU forward batches were {observed}; expected {expected}')
    if not np.isfinite(sums).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError('Non-finite logits or uncovered voxels after patch stitching')
    logits = sums / weights[None]
    if not np.isfinite(logits).all():
        raise RuntimeError('Non-finite averaged logits')
    labels = predictor.label_manager.convert_logits_to_segmentation(logits)
    if tuple(labels.shape) != shape:
        raise RuntimeError('Final segmentation has the wrong shape')

    print('\nPASS: real model inputs AND outputs were on CUDA.', flush=True)
    print(f'Observed batch sizes at the network: {observed}', flush=True)
    print(f'Patch size: {patch}; finite logits, full coverage and final label shape verified.', flush=True)
    print(f'Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB', flush=True)
    print(f'Prediction + stitching: {elapsed:.1f}s; total including imports/model load: {perf_counter() - start:.1f}s', flush=True)
    print(f'Mirroring: {args.mirroring}; torch.compile: disabled. No Zarr input/output was used.', flush=True)
    print('This verifies GPU batching, not real-image preprocessing, segmentation quality or full-volume speed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'\nFAIL: {type(error).__name__}: {error}', flush=True)
        print('If GPU memory ran out, retry with --batch-size 2. Otherwise inspect the error above.', flush=True)
        raise SystemExit(1)

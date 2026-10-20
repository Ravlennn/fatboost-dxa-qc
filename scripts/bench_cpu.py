"""CPU timing of the release model on synthetic images: no patient data, so it can run on any machine.

Every branch is forced (lumbar spine, right hip, left hip) instead of relying on the region classifier,
which would route arbitrary synthetic pixels to an arbitrary branch. Image sizes match the organisers'
test images (spine 317×300, hip 291×280). The network inputs have fixed sizes, so the timing does not
depend on what the pixels show; scripts/bench_cpu.py run next to real images checks that on one machine.

    python scripts/bench_cpu.py [--repeats 5] [--threads 2] [--json out.json]
In the image:  docker run --rm --cpus=2 --memory=4g --network none -v "$PWD/scripts:/bench:ro" \
                 --entrypoint python dxaqc:local /bench/bench_cpu.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

SIZES = {'lumbar_spine': (317, 300), 'hip_right': (291, 280), 'hip_left': (291, 280)}


def synthetic(shape, seed):
    """A bright vertical band (bone) on a dark field with soft noise: plausible intensities, no patient data."""
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    band = np.exp(-((x - w / 2) / (w / 7)) ** 2) * (0.6 + 0.4 * np.sin(y / 18) ** 2)
    img = 30 + 180 * band + rng.normal(0, 8, shape)
    return np.clip(img, 0, 252).astype(np.uint8)


def one_image(predictor, pixels, region):
    """What process_image does for one file: embed, then the region's heads, landmarks, detectors and CNNs."""
    features = predictor.encoder.embed(pixels)
    return predictor._predict(pixels, region, features, mm_per_px=0.6)


def cpu_name():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--model-dir')
    parser.add_argument('--json')
    args = parser.parse_args()

    from dxaqc.api import create_predictor
    t0 = time.perf_counter()
    predictor = create_predictor(args.model_dir, threads=args.threads)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    predictor.warm_up()
    t_warm = time.perf_counter() - t0

    result = {'machine': platform.machine(), 'cpu': cpu_name(), 'os_cpus': os.cpu_count(), 'threads': args.threads,
              'load_s': round(t_load, 2), 'warm_up_s': round(t_warm, 2), 'regions': {}}
    for k, (region, shape) in enumerate(SIZES.items()):
        pixels = synthetic(shape, k)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            one_image(predictor, pixels, region)
            times.append(time.perf_counter() - t0)
        result['regions'][region] = {'median_s': round(statistics.median(times), 3), 'max_s': round(max(times), 3),
                                     'runs': [round(t, 3) for t in times]}
    med = {r: v['median_s'] for r, v in result['regions'].items()}
    # a study in the organisers' set: one spine and both hips, after the models are loaded
    result['study_spine_2hips_s'] = round(med['lumbar_spine'] + med['hip_right'] + med['hip_left'], 2)
    result['study_with_model_load_s'] = round(result['study_spine_2hips_s'] + t_load + t_warm, 2)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()

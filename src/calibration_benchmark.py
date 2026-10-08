"""CPU topic A benchmark. Run from repo root; never reuses an output directory.

Score uses only candidate projections and image edges, never a reference calibration
or reference UV. Fixed random point IDs are selected from finite input XYZ before
projection. Only candidate-visible samples enter the median; coverage is reported.
This heuristic can miss drift on textureless/repetitive scenes or changing FOV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.projection import velo_to_cam, cam_to_image, perturb_extrinsic

ANGLES = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
TRANSLATIONS = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]
ROOTS = ['data/kitti_mini', 'data/nuscenes_mini_subset', 'data/synthetic']
MIN_SCORE_VISIBLE = 30  # Fixed before experiments; never selected using test data.
NATIVE_AXES = {
    'kitti_mini': 'LiDAR native: x forward, y left, z up',
    'synthetic': 'LiDAR native: x forward, y left, z up (KITTI convention)',
    'nuscenes_mini_subset': 'LiDAR native: x right, y forward, z up',
}
CAVEATS = [
    'nuScenes 2D bbox is generated from 3D labels, not independent 2D ground truth.',
    'nuScenes KITTI-style upright box conversion is approximate (full orientation is lost).',
    'Ego motion compensation is enabled; dynamic object motion between sensor timestamps is not compensated.',
    'Membership is fixed per baseline object; overlapping 3D boxes may share points.',
    'Frame split is within-seen-scenes only, deterministic and stratified by dataset/scene; adjacent frames are correlated. No unseen-scene generalization claim.',
    'Synthetic calibration perturbations measure sensitivity, not validated real-world drift detection.',
    'Canny score median uses candidate-visible fixed samples; changing coverage can bias it. Lower is better.',
    'Canny heuristic matches projected points to image edges, not depth-edge to image-edge alignment; not a validated alignment metric.',
    'Fewer than 30 candidate-visible fixed samples: coverage_insufficient, unscorable; fixed rule, no test tuning.',
    'Perturbation axes are LiDAR native. Yaw is about z-up in both datasets; roll/pitch and x/y translations do not represent the same physical direction across KITTI and nuScenes.',
    'Zero is an unperturbed control, not proof that dataset calibration/labels are perfect.',
]


def configurations():
    out = [dict(config='zero', factor='zero', value=0.0, unit='none')]
    for factor in ['yaw', 'pitch', 'roll', 'x', 'y', 'z']:
        for value in ANGLES if factor in ['yaw', 'pitch', 'roll'] else TRANSLATIONS:
            if value != 0:
                out.append(dict(config=f'{factor}_{value:+g}', factor=factor,
                                value=float(value), unit='deg' if factor in ['yaw', 'pitch', 'roll'] else 'm'))
    return out


def drift(calib, cfg):
    if cfg['factor'] == 'zero':
        return calib
    factor, value = cfg['factor'], cfg['value']
    if factor in ['yaw', 'pitch', 'roll']:
        return perturb_extrinsic(calib, **{factor + '_deg': value})
    t = [0.0, 0.0, 0.0]
    t[['x', 'y', 'z'].index(factor)] = value
    return perturb_extrinsic(calib, t_xyz_m=tuple(t))


def project(points, calib, shape):
    uv, depth, mask = cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2, shape)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(points),) or len(uv) != mask.sum():
        raise ValueError('Projection must preserve input order and return an N-length boolean mask.')
    if not np.isfinite(uv).all() or not np.isfinite(depth).all():
        raise ValueError('Projection returned nonfinite valid UV/depth.')
    dense = np.full((len(points), 2), np.nan)
    dense[mask] = uv
    return dense, mask, np.asarray(depth)


def membership(points_cam, obj):
    """Inverse KITTI yaw, bottom center: x ±l/2, y [-h,0], z ±w/2."""
    h, w, length = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ rotation
    return (np.isfinite(local).all(1) & (np.abs(local[:, 0]) <= length / 2)
            & (local[:, 1] >= -h) & (local[:, 1] <= 0)
            & (np.abs(local[:, 2]) <= w / 2))


def edge_map(image):
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    # No edges: normalized distance 1, not a misleading perfect score.
    if not edges.any():
        return np.full(edges.shape, np.hypot(*edges.shape), np.float32)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def edge_score(dense, valid, ids, distances):
    selected = ids[valid[ids]]
    if len(selected) < MIN_SCORE_VISIBLE:
        return float('nan'), len(selected)
    pixels = np.floor(dense[selected]).astype(int)
    values = distances[pixels[:, 1], pixels[:, 0]] / np.hypot(*distances.shape)
    return float(np.median(values)), len(selected)


def split_frames(dataset, frames, seed):
    result = {}
    groups = {}
    for frame in frames:
        scene = frame.rsplit('_', 1)[0] if dataset == 'nuscenes_mini_subset' else dataset
        groups.setdefault(scene, []).append(frame)
    for scene, group in sorted(groups.items()):
        digest = hashlib.sha256(f'{seed}:{dataset}:{scene}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        order = rng.permutation(sorted(group))
        n = max(1, min(len(group) - 1, len(group) // 2)) if len(group) > 1 else 1
        result.update({str(frame): 'calibration' if i < n else 'test' for i, frame in enumerate(order)})
    return result


def environment():
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    return dict(python=sys.version, platform=platform.platform(), cpu=cpu,
                logical_cpus=os.cpu_count(), numpy=np.__version__, opencv=cv2.__version__,
                pandas=pd.__version__, opencv_threads=cv2.getNumThreads(),
                code_sha256=code_hashes(),
                projection_sha256=file_sha256(Path('starter/projection.py')),
                thread_environment={key: os.environ.get(key) for key in
                                    ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes():
    paths = ['src/calibration_benchmark.py', 'starter/projection.py',
             'starter/datasets.py', 'starter/kitti_io.py', 'starter/nuscenes_io.py']
    return {path: file_sha256(path) for path in paths}


def data_manifest(datasets):
    return [dict(dataset=root.name, path=str(path), relative_path=str(path.relative_to(root)),
                 bytes=path.stat().st_size, sha256=file_sha256(path))
            for root, _ in datasets for path in sorted(root.rglob('*')) if path.is_file()]


def write_json(path, value):
    # Exclusive creation also protects individual artifacts.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def json_records(table):
    return json.loads(table.to_json(orient='records'))


def csv_payloads(table, filename, max_bytes=20_000_000, shard_rows=20000):
    """Preserve the unsuffixed API below the limit; UTF-8 bytes strictly <20 MB."""
    payload = table.to_csv(index=False).encode('utf-8')
    if len(payload) < max_bytes:
        yield filename, len(table), payload
        return
    del payload
    parts = []
    def divide(chunk):
        data = chunk.to_csv(index=False).encode('utf-8')
        if len(data) < max_bytes:
            parts.append((len(chunk), data))
        elif len(chunk) > 1:
            mid = len(chunk) // 2
            divide(chunk.iloc[:mid])
            divide(chunk.iloc[mid:])
        else:
            raise ValueError('Single CSV row/header exceeds byte limit; cannot shard safely.')
    for start in range(0, len(table), shard_rows):
        divide(table.iloc[start:start + shard_rows])
    stem = Path(filename).stem
    for index, (rows, data) in enumerate(parts):
        yield f'{stem}_{index:03d}.csv', rows, data


def write_csv(table, path):
    artifacts = []
    for filename, rows, payload in csv_payloads(table, path.name):
        with (path.parent / filename).open('xb') as stream:
            stream.write(payload)
        artifacts.append(dict(file=filename, rows=rows, bytes=len(payload),
                              sha256=hashlib.sha256(payload).hexdigest()))
    return artifacts


def summary(frames, objects):
    rows = []
    for (dataset, config), group in frames.groupby(['dataset', 'config'], sort=False):
        ob = objects[(objects.dataset == dataset) & (objects.config == config)] if len(objects) else objects
        eligible = ob[ob.eligible] if len(ob) else ob
        n, common = group.n_points.sum(), group.common_valid.sum()
        den = eligible.baseline_visible_assigned.sum() if len(eligible) else 0
        rows.append(dict(dataset=dataset, config=config, factor=group.factor.iloc[0],
                         value=group.value.iloc[0], unit=group.unit.iloc[0], frames=len(group),
                         fov_micro=group.inside_count.sum() / n if n else np.nan,
                         fov_macro=group.inside_fov.mean(),
                         displacement_px_micro=group.displacement_sum_px.sum() / common if common else np.nan,
                         displacement_px_macro=group.displacement_mean_px.mean(),
                         coverage_micro=common / group.baseline_inside_count.sum() if group.baseline_inside_count.sum() else np.nan,
                         coverage_macro=group.common_coverage.mean(),
                         association_micro=eligible.bbox_hits.sum() / den if den else np.nan,
                         association_macro=eligible.association_ratio.mean() if len(eligible) else np.nan,
                         eligible_objects=len(eligible), edge_score_macro=group.edge_score.mean(),
                         score_coverage_micro=group.score_visible_count.sum() / group.score_sample_count.sum()
                         if group.score_sample_count.sum() else np.nan,
                         score_coverage_macro=group.score_coverage.mean()))
    return pd.DataFrame(rows)


def detection_rates(group, threshold):
    """Unscorable test rows count as misses; conditional recall is explicitly separate."""
    available = group[group.score_available]
    threshold_available = threshold is not None
    rate = float(group.detected.mean()) if threshold_available and len(group) else None
    conditional = float(available.detected.mean()) if threshold_available and len(available) else None
    return dict(threshold=threshold, threshold_available=threshold_available,
                unavailable_reason=None if threshold_available else 'No scorable zero calibration frames (minimum visible samples = 30).',
                n=len(group), n_score_available=len(available),
                unscorable_rate=float((~group.score_available).mean()) if len(group) else None,
                sensitivity_recall=rate, sensitivity_recall_conditional=conditional,
                missed_or_unscorable=int((~group.detected).sum()) if threshold_available else None)


def detection(frames, out):
    """95th percentile train zero only; test is never used to choose threshold."""
    evaluations, roc_rows, thresholds = [], [], {}
    for dataset, group in frames.groupby('dataset'):
        train = group[(group.split == 'calibration') & (group.config == 'zero')].edge_score.dropna()
        threshold = float(np.quantile(train, .95)) if len(train) else None
        thresholds[dataset] = threshold
        test = group[group.split == 'test'].copy()
        test['threshold'] = threshold
        test['threshold_available'] = threshold is not None
        test['score_available'] = test.edge_score.notna()
        test['detected'] = test.edge_score > threshold if threshold is not None else False
        test['is_perturbed'] = test.config != 'zero'
        evaluations.append(test)
        finite = test[test.score_available]
        negative, positive = finite[~finite.is_perturbed], finite[finite.is_perturbed]
        candidates = np.r_[-np.inf, np.unique(finite.edge_score), np.inf]
        for cut in candidates:
            roc_rows.append(dict(dataset=dataset, threshold=cut,
                                 fpr=float((negative.edge_score > cut).mean()) if len(negative) else np.nan,
                                 recall=float((positive.edge_score > cut).mean()) if len(positive) else np.nan))
    evaluation = pd.concat(evaluations, ignore_index=True)
    write_csv(evaluation, out / 'score_test.csv')
    roc = pd.DataFrame(roc_rows)
    write_csv(roc, out / 'score_roc.csv')
    reports = []
    for (dataset, factor, value), group in evaluation.groupby(['dataset', 'factor', 'value']):
        available = group[group.score_available]
        rates = detection_rates(group, thresholds[dataset])
        rates['false_positive_rate'] = rates['sensitivity_recall'] if factor == 'zero' else None
        rates['false_positive_rate_conditional'] = rates['sensitivity_recall_conditional'] if factor == 'zero' else None
        if factor == 'zero':
            rates['sensitivity_recall'] = rates['sensitivity_recall_conditional'] = None
        reports.append(dict(dataset=dataset, factor=factor, value=float(value), **rates))
    overall = [dict(dataset=dataset, **detection_rates(group[group.is_perturbed], thresholds[dataset]))
               for dataset, group in evaluation.groupby('dataset')]
    write_json(out / 'score_evaluation.json', dict(threshold_rule='baseline calibration quantile 0.95; strict score > threshold',
               thresholds=thresholds, results=reports, overall_test_drift=overall, caveats=CAVEATS,
               min_score_visible=MIN_SCORE_VISIBLE,
               interpretation='Report all rates as observed; no test tuning. Recall is synthetic sensitivity only.'))
    return evaluation, roc, thresholds


def plots(table, evaluation, roc, thresholds, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset, group in table.groupby('dataset'):
        metrics = [('fov_micro', 'Inside FOV / all input points'),
                   ('displacement_px_micro', 'Common-valid displacement (px)'),
                   ('association_micro', 'Correct bbox / baseline-visible assigned'),
                   ('coverage_micro', 'Common valid / baseline valid'),
                   ('edge_score_macro', 'Median edge distance / image diagonal'),
                   ('score_coverage_micro', 'Visible score samples / fixed samples')]
        for kind, factors, unit in [('angles', ['yaw', 'pitch', 'roll'], 'degrees'),
                                    ('translation', ['x', 'y', 'z'], 'meters')]:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, (metric, label) in zip(axes.flat, metrics):
                baseline = group[group.config == 'zero']
                for factor in factors:
                    curve = pd.concat([group[group.factor == factor], baseline]).sort_values('value')
                    ax.plot(curve.value, curve[metric], '.-', label=factor)
                ax.set(title=label, xlabel=f'Independent perturbation ({unit})')
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
            fig.suptitle(dataset + f' — {kind}; micro metrics, edge score frame macro')
            fig.text(.5, .01, NATIVE_AXES.get(dataset, 'LiDAR native axes: verify custom dataset convention')
                     + ' | roll=x, pitch=y, yaw=z; x/y and roll/pitch are not physically equivalent across datasets',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .035, 1, .96))
            fig.savefig(out / f'sweep_{dataset}_{kind}.png', dpi=140)
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dataset, group in roc.groupby('dataset'):
        axes[0].plot(group.fpr, group.recall, label=dataset)
        test = evaluation[(evaluation.dataset == dataset) & evaluation.score_available]
        for perturbed, symbol in [(False, 'o'), (True, 'x')]:
            data = test[test.is_perturbed == perturbed]
            axes[1].scatter(np.full(len(data), list(thresholds).index(dataset) + .15 * perturbed),
                            data.edge_score, s=8, marker=symbol,
                            label=f'{dataset}: ' + ('perturbed' if perturbed else 'zero'))
        if thresholds[dataset] is not None:
            i = list(thresholds).index(dataset)
            axes[1].plot([i - .2, i + .35], [thresholds[dataset]] * 2, 'k-')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=.3)
    axes[0].set(xlabel='Test zero false-positive rate (scorable only)', ylabel='Conditional recall (scorable only)', title='ROC: descriptive; see JSON for all-test recall')
    axes[1].set(xticks=range(len(thresholds)), xticklabels=list(thresholds), ylabel='Normalized edge distance', title='Calibration-only thresholds (black)')
    for ax in axes:
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'score_roc_threshold.png', dpi=140)
    plt.close(fig)


def overlay(frame, cfg, title, max_points=6000, selected_object_id=None):
    image = frame['image'].copy()
    dense, valid, depth = project(frame['points'], drift(frame['calib'], cfg), image.shape)
    ids = np.flatnonzero(valid)
    step = max(1, int(np.ceil(len(ids) / max_points)))
    ids = ids[::step]
    # Rendering is capped at 6000 circles per panel; full clouds are used for metrics.
    if len(ids):
        pixels = np.floor(dense[ids]).astype(int)
        color = cv2.applyColorMap((255 * (1 - np.clip(depth[::step] / 70, 0, 1))).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, 0]
        for (u, v), bgr in zip(pixels, color):
            cv2.circle(image, (int(u), int(v)), 2, tuple(int(c) for c in bgr), -1)
    for object_id, obj in enumerate(frame['labels']):
        x1, y1, x2, y2 = np.round(obj.bbox).astype(int)
        distance = float(np.linalg.norm(obj.location))
        selected = object_id == selected_object_id
        box_color = (0, 255, 255) if selected else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3 if selected else 1)
        prefix = 'SELECTED ' if selected else ''
        cv2.putText(image, f'{prefix}#{object_id} {obj.type} {distance:.2f}m',
                    (max(0, x1), max(45, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5 if selected else .42, box_color, 2 if selected else 1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return image


def save_image(path, image):
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def visual_artifacts(catalog, frames, evaluation, configs, out):
    zero = configs[0]
    manifest, used = [], set()
    kitti = [item for item in catalog if item['dataset'] == 'kitti_mini' and item['object_distances_m']]
    for band, target in [('near', 5), ('mid', 25), ('far', 60)]:
        candidates = [item for item in kitti if item['frame'] not in used]
        if not candidates:
            raise RuntimeError('Need three distinct labeled KITTI frames for near/mid/far overlays.')
        _, _, object_id, item = min(
            (abs(distance - target), item['frame'], object_id, item)
            for item in candidates for object_id, distance in enumerate(item['object_distances_m']))
        actual_distance = item['object_distances_m'][object_id]
        used.add(item['frame'])
        frame = load_frame(item['root'], item['frame'])
        name = f'overlay_kitti_{band}_{item["frame"]}.png'
        save_image(out / name, overlay(frame, zero,
                   f'KITTI {band} | {item["frame"]} | selected #{object_id}: {actual_distance:.2f}m',
                   selected_object_id=object_id))
        manifest.append(dict(file=name, selection_target_m=target, selected_object_id=object_id,
                             selected_object_type=frame['labels'][object_id].type,
                             selected_actual_distance_m=actual_distance,
                             distance_convention='Euclidean norm of 3D bottom-center in baseline rectified camera frame',
                             selection_rule='Closest object to target among frames not selected previously; no fixed distance bin claimed.',
                             **item))
    for scene, light in [('scene-0103', 'day'), ('scene-1094', 'night')]:
        items = [item for item in catalog if item['dataset'] == 'nuscenes_mini_subset' and item['frame'].startswith(scene)]
        if not items:
            raise RuntimeError(f'Missing required nuScenes {light} scene {scene}')
        item = items[len(items) // 2]
        name = f'overlay_nuscenes_{light}_{item["frame"]}.png'
        save_image(out / name, overlay(load_frame(item['root'], item['frame']), zero,
                                       f'nuScenes {light} | {item["frame"]} | approx labels'))
        manifest.append(dict(file=name, **item))
    write_json(out / 'overlay_manifest.json', manifest)
    # Prefer strongest missed test drift; otherwise strongest displacement with FOV delta <=1 pp.
    candidate = evaluation[(evaluation.config != 'zero') & (~evaluation.detected)
                           & (evaluation.dataset != 'synthetic') & (evaluation.displacement_mean_px > 0)].copy()
    reason = 'Test score misses synthetic drift (or score unavailable); largest common-valid mean displacement.'
    if candidate.empty:
        candidate = frames[(frames.config != 'zero') & (frames.dataset != 'synthetic')
                           & (frames.fov_change.abs() <= .01)].copy()
        reason = 'Inside FOV changes at most 1 percentage point; largest common-valid mean displacement.'
    if candidate.empty:
        raise RuntimeError('No qualifying objective failure case; artifacts incomplete, do not invent a failure.')
    row = candidate.sort_values(['displacement_mean_px', 'dataset', 'frame', 'config'], ascending=[False, True, True, True]).iloc[0]
    item = next(x for x in catalog if x['dataset'] == row.dataset and x['frame'] == row.frame)
    frame = load_frame(item['root'], item['frame'])
    cfg = next(c for c in configs if c['config'] == row.config)
    left = overlay(frame, zero, f'Baseline | {row.frame}')
    right = overlay(frame, cfg, f'{row.config} | displacement={row.displacement_mean_px:.2f}px')
    name = f'fail_01_{row.dataset}_{row.frame}_{row.config}.png'
    save_image(out / name, np.hstack([left, right]))
    write_json(out / 'fail_01_facts.json', dict(file=name, selection_rule=reason,
               metrics=json_records(pd.DataFrame([row]))[0], source=item, caveats=CAVEATS))


def latency(frame, dataset, repeats, out_rows):
    points, calib, shape = frame['points'], frame['calib'], frame['image'].shape
    for _ in range(5):
        project(points, calib, shape)
    samples = []
    for i in range(repeats):
        start = time.perf_counter_ns()
        # Time only starter projection, not IO, drift construction, metrics or dense UV allocation.
        cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2, shape)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        samples.append(elapsed)
        out_rows.append(dict(dataset=dataset, frame=frame['frame_id'], repeat=i, milliseconds=elapsed,
                             n_points=len(points)))
    return dict(dataset=dataset, frame=frame['frame_id'], repeats=repeats, warmup=5,
                p50_ms=float(np.percentile(samples, 50)), p95_ms=float(np.percentile(samples, 95)),
                scope='velo_to_cam + cam_to_image; preloaded same input; CPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', required=True, type=Path, help='New artifact directory, e.g. results/topic_a_20261007_01. Existing directories are refused.')
    parser.add_argument('--data-roots', nargs='+', default=ROOTS, help='Default: all 20 KITTI, 80 nuScenes and 5 synthetic frames. No frame truncation.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--score-samples', type=int, default=4096, help='Fixed finite XYZ point IDs/frame, before any projection.')
    parser.add_argument('--latency-repeats', type=int, default=30, help='At least 20, after five warmups.')
    args = parser.parse_args()
    if args.score_samples < 1 or args.latency_repeats < 20:
        parser.error('score-samples >=1 and latency-repeats >=20 required')
    if args.out_dir.exists():
        parser.error('Output directory already exists; choose a fresh runid. Nothing overwritten.')
    datasets = [(Path(root), list_frames(root)) for root in args.data_roots]
    if len({root.name for root, _ in datasets}) != len(datasets):
        parser.error('Dataset root basenames must be unique.')
    if any(not ids for _, ids in datasets):
        parser.error('Every dataset must contain frames.')
    # Fail before creating artifacts when CP2 is unfinished.
    first = load_frame(datasets[0][0], datasets[0][1][0])
    project(first['points'], first['calib'], first['image'].shape)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir
    cv2.setNumThreads(1)
    env = environment()
    configs = configurations()
    manifest = data_manifest(datasets)
    write_json(out / 'data_manifest.json', manifest)
    split_rows = [dict(dataset=root.name, frame=frame, split=split,
                       scene=frame.rsplit('_', 1)[0] if root.name == 'nuscenes_mini_subset' else root.name)
                  for root, ids in datasets for frame, split in split_frames(root.name, ids, args.seed).items()]
    write_csv(pd.DataFrame(split_rows), out / 'split_manifest.csv')
    write_json(out / 'split_manifest.json', split_rows)
    env.update(data_manifest='data_manifest.json',
               data_manifest_sha256=file_sha256(out / 'data_manifest.json'),
               split_manifest='split_manifest.json', split_manifest_csv='split_manifest.csv',
               split_manifest_sha256=file_sha256(out / 'split_manifest.json'))
    write_json(out / 'environment.json', env)
    write_json(out / 'config.json', dict(seed=args.seed, roots=[str(r) for r, _ in datasets],
               frames={r.name: ids for r, ids in datasets}, configurations=configs,
               sweep_zero='One shared zero control included in every factor curve; factors never combined.',
               angles_deg=ANGLES, translations_m=TRANSLATIONS, score_samples=args.score_samples,
               canny=[100, 200], score='median candidate-visible edge distance / image diagonal',
               min_score_visible=MIN_SCORE_VISIBLE, score_insufficient_status='coverage_insufficient',
               native_axes=NATIVE_AXES, factor_axes=dict(roll='native x', pitch='native y', yaw='native z',
                                                       x='native x', y='native y', z='native z'),
               code_sha256=env['code_sha256'], data_manifest='data_manifest.json',
               data_manifest_sha256=file_sha256(out / 'data_manifest.json'),
               split_manifest='split_manifest.json', split_manifest_csv='split_manifest.csv',
               split_manifest_sha256=file_sha256(out / 'split_manifest.json'),
               min_object_baseline_visible=5, latency_repeats=args.latency_repeats, caveats=CAVEATS))
    frame_rows, object_rows, catalog, latency_rows, latency_stats = [], [], [], [], []
    for root, ids in datasets:
        dataset = root.name
        splits = split_frames(dataset, ids, args.seed)
        for index, frame_id in enumerate(ids):
            frame = load_frame(root, frame_id)
            points, calib, image = frame['points'], frame['calib'], frame['image']
            baseline_cam = velo_to_cam(points[:, :3], calib)
            base_uv, base_valid, _ = project(points, calib, image.shape)
            assigned = [np.flatnonzero(membership(baseline_cam, obj) & base_valid) for obj in frame['labels']]
            digest = hashlib.sha256(f'{args.seed}:{dataset}:{frame_id}:score'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
            finite = np.flatnonzero(np.isfinite(points[:, :3]).all(1))
            sample_ids = np.sort(rng.choice(finite, min(len(finite), args.score_samples), replace=False))
            distances = edge_map(image)
            object_distances = [float(np.linalg.norm(obj.location)) for obj in frame['labels']]
            catalog.append(dict(dataset=dataset, root=str(root), frame=frame_id, split=splits[frame_id],
                                object_distances_m=object_distances))
            for cfg in configs:
                uv, valid, _ = (base_uv, base_valid, None) if cfg['config'] == 'zero' else project(points, drift(calib, cfg), image.shape)
                common = base_valid & valid
                displacement = np.linalg.norm(uv[common] - base_uv[common], axis=1)
                score, score_visible = edge_score(uv, valid, sample_ids, distances)
                hits_all, den_all, ratios = 0, 0, []
                for j, (obj, members) in enumerate(zip(frame['labels'], assigned)):
                    visible_ids = members[valid[members]]
                    pixels = uv[visible_ids]
                    x1, y1, x2, y2 = obj.bbox
                    hits = int(((pixels[:, 0] >= x1) & (pixels[:, 0] <= x2)
                                & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)).sum())
                    eligible = len(members) >= 5
                    ratio = hits / len(members) if eligible else np.nan
                    if eligible:
                        hits_all += hits
                        den_all += len(members)
                        ratios.append(ratio)
                    object_rows.append(dict(dataset=dataset, frame=frame_id, split=splits[frame_id], **cfg,
                        object_id=j, object_type=obj.type, distance_m=object_distances[j], eligible=eligible,
                        baseline_visible_assigned=len(members), perturbed_visible_assigned=len(visible_ids),
                        offscreen_misses=len(members) - len(visible_ids), bbox_hits=hits, association_ratio=ratio,
                        bbox_x1=float(x1), bbox_y1=float(y1), bbox_x2=float(x2), bbox_y2=float(y2)))
                n, base_n = len(points), int(base_valid.sum())
                frame_rows.append(dict(dataset=dataset, frame=frame_id, split=splits[frame_id], **cfg,
                    n_points=n, invalid_xyz=int((~np.isfinite(points[:, :3]).all(1)).sum()),
                    baseline_inside_count=base_n, inside_count=int(valid.sum()), inside_fov=float(valid.sum() / n) if n else np.nan,
                    fov_change=float((valid.sum() - base_n) / n) if n else np.nan,
                    common_valid=int(common.sum()), common_coverage=float(common.sum() / base_n) if base_n else np.nan,
                    displacement_sum_px=float(displacement.sum()), displacement_mean_px=float(displacement.mean()) if len(displacement) else np.nan,
                    displacement_median_px=float(np.median(displacement)) if len(displacement) else np.nan,
                    displacement_p95_px=float(np.percentile(displacement, 95)) if len(displacement) else np.nan,
                    bbox_hits=hits_all, baseline_visible_assigned=den_all,
                    association_micro=hits_all / den_all if den_all else np.nan,
                    association_macro=float(np.mean(ratios)) if ratios else np.nan,
                    edge_score=score, score_sample_count=len(sample_ids), score_visible_count=score_visible,
                    score_status='ok' if score_visible >= MIN_SCORE_VISIBLE else 'coverage_insufficient',
                    score_coverage=score_visible / len(sample_ids) if len(sample_ids) else np.nan,
                    sensor_time_delta_ms=(frame['timestamp_camera_us'] - frame['timestamp_lidar_us']) / 1000 if 'timestamp_camera_us' in frame else np.nan))
            if index == 0 and dataset in ['kitti_mini', 'nuscenes_mini_subset']:
                latency_stats.append(latency(frame, dataset, args.latency_repeats, latency_rows))
            print(f'{dataset} {index + 1}/{len(ids)} {frame_id}', flush=True)
    frames, objects = pd.DataFrame(frame_rows), pd.DataFrame(object_rows)
    frame_files = write_csv(frames, out / 'per_frame_config.csv')
    object_files = write_csv(objects, out / 'per_object_config.csv')
    write_json(out / 'csv_manifest.json', dict(per_frame_config=frame_files,
               per_object_config=object_files, byte_limit_exclusive=20_000_000,
               target_shard_rows=20000, aggregation='summary uses the entire DataFrame, not a single shard'))
    aggregate = summary(frames, objects)
    write_csv(aggregate, out / 'summary.csv')
    write_csv(pd.DataFrame(latency_rows), out / 'latency_repeats.csv')
    write_csv(pd.DataFrame([{**row, 'cpu': env['cpu'], 'platform': env['platform'], 'opencv_threads': env['opencv_threads']} for row in latency_stats]), out / 'latency_summary.csv')
    evaluation, roc, thresholds = detection(frames, out)
    plots(aggregate, evaluation, roc, thresholds, out)
    visual_artifacts(catalog, frames, evaluation, configs, out)
    write_json(out / 'completion.json', dict(status='complete', frame_configs=len(frames),
               object_configs=len(objects), elapsed_scope='No training; all listed frames evaluated.',
               caveats=CAVEATS))
    print(f'Complete: {out}. Read score_evaluation.json and fail_01_facts.json before making claims.')


if __name__ == '__main__':
    main()

"""Read-only tests: python -B -m unittest discover -s src/tests -v."""
import unittest
from unittest.mock import patch
from io import StringIO

import numpy as np
import pandas as pd

from starter.kitti_io import load_calib, KittiCalib, KittiObject
from starter.projection import velo_to_cam, cam_to_image
from src.calibration_benchmark import (membership, configurations, split_frames, edge_score,
                                       overlay, detection_rates, csv_payloads)


class ProjectionTests(unittest.TestCase):
    def test_known_synthetic_point(self):
        calib = load_calib('data/synthetic/training/calib/000000.txt')
        cam = velo_to_cam(np.array([[10., 0., 0.]]), calib)
        self.assertAlmostEqual(cam[0, 2], 9.72732109, places=6)
        uv, depth, mask = cam_to_image(cam, calib.P2, (375, 1242, 3))
        np.testing.assert_allclose(uv, [[613.96414869, 175.00653723]], atol=1e-6)
        np.testing.assert_allclose(depth, [9.72732109], atol=1e-6)
        np.testing.assert_array_equal(mask, [True])

    def test_depth_nonfinite_and_image_bounds(self):
        points = np.array([[1., 1., 1.], [0., 0., 1.], [9.999, 9.999, 1.],
                           [10., 0., 1.], [0., 10., 1.], [-.01, 0., 1.],
                           [0., -.01, 1.], [0., 0., -1.], [0., 0., 0.],
                           [0., 0., .1], [.02, .02, .2],
                           [np.nan, 0., 1.], [0., np.inf, 1.], [0., 0., np.inf]])
        P = np.column_stack([np.eye(3), np.zeros(3)])
        uv, depth, mask = cam_to_image(points, P, (10, 10, 3))
        np.testing.assert_array_equal(mask, [True, True, True, False, False, False,
                                           False, False, False, False, True, False, False, False])
        np.testing.assert_allclose(uv, [[1., 1.], [0., 0.], [9.999, 9.999], [.1, .1]])
        np.testing.assert_allclose(depth, [1., 1., 1., .2])

    def test_empty_and_all_invalid(self):
        P = np.column_stack([np.eye(3), np.zeros(3)])
        for points in [np.empty((0, 3)), np.array([[0., 0., -1.]])]:
            uv, depth, mask = cam_to_image(points, P, (10, 10))
            self.assertEqual(uv.shape, (0, 2))
            self.assertEqual(depth.shape, (0,))
            self.assertEqual(mask.shape, (len(points),))
            self.assertFalse(mask.any())

    def test_zero_projective_denominator(self):
        P = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]])
        uv, depth, mask = cam_to_image(np.array([[0., 0., 1.], [1., 1., 1.]]), P, (10, 10))
        self.assertFalse(mask.any())
        self.assertEqual(uv.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))

    def test_transform_direction(self):
        transform = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., 4.]])
        calib = KittiCalib(np.zeros((3, 4)), np.eye(3), transform)
        np.testing.assert_allclose(velo_to_cam(np.array([[1., 2., 3.]]), calib), [[0., 4., 7.]])

    def test_calibration_must_be_finite(self):
        points = np.array([[1., 1., 1.]])
        for bad in [np.nan, np.inf]:
            P = np.column_stack([np.eye(3), np.zeros(3)])
            P[0, 0] = bad
            with self.assertRaises(ValueError):
                cam_to_image(points, P, (10, 10))
            transform = np.column_stack([np.eye(3), np.zeros(3)])
            transform[0, 0] = bad
            with np.errstate(invalid='ignore'), self.assertRaises(ValueError):
                velo_to_cam(points, KittiCalib(np.zeros((3, 4)), np.eye(3), transform))
            rect = np.eye(3)
            rect[0, 0] = bad
            with np.errstate(invalid='ignore'), self.assertRaises(ValueError):
                velo_to_cam(points, KittiCalib(np.zeros((3, 4)), rect,
                                             np.column_stack([np.eye(3), np.zeros(3)])))

    def test_image_dimensions_must_be_positive(self):
        P = np.column_stack([np.eye(3), np.zeros(3)])
        for shape in [(0, 10), (10, 0), (-1, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                cam_to_image(np.array([[1., 1., 1.]]), P, shape)


class BenchmarkContractTests(unittest.TestCase):
    def test_inverse_yaw_bottom_center(self):
        obj = KittiObject('Car', 0., 0, 0., np.zeros(4), np.array([2., 2., 4.]),
                          np.array([3., 5., 10.]), np.pi / 2)
        local = np.array([[1.9, -1., .9], [2.1, -1., 0.], [0., .01, 0.],
                          [0., -2.01, 0.], [0., -1., 1.01], [np.nan, 0., 0.]])
        c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        camera = local @ rotation.T + obj.location
        np.testing.assert_array_equal(membership(camera, obj), [True, False, False, False, False, False])

    def test_independent_sweep_and_deterministic_split(self):
        configs = configurations()
        self.assertEqual(len(configs), 43)
        self.assertEqual(sum(c['config'] == 'zero' for c in configs), 1)
        frames = [f'scene-0103_{i:03d}' for i in range(40)]
        first = split_frames('nuscenes_mini_subset', frames, 2026)
        self.assertEqual(first, split_frames('nuscenes_mini_subset', frames[::-1], 2026))
        self.assertEqual(sum(v == 'test' for v in first.values()), 20)

    def test_edge_score_fixed_ids_and_visibility(self):
        dense = np.repeat([[1., 1.]], 31, axis=0)
        dense[-1] = np.nan
        distances = np.arange(16, dtype=float).reshape(4, 4)
        valid = np.arange(31) < 30
        score, count = edge_score(dense, valid, np.arange(31), distances)
        self.assertEqual(count, 30)
        self.assertAlmostEqual(score, 5 / np.hypot(4, 4))
        score, count = edge_score(dense, valid, np.arange(29), distances)
        self.assertEqual(count, 29)
        self.assertTrue(np.isnan(score))

    def test_recall_includes_unscorable_and_unavailable_threshold(self):
        group = pd.DataFrame(dict(score_available=[True, True, False], detected=[True, False, False]))
        rates = detection_rates(group, .1)
        self.assertAlmostEqual(rates['sensitivity_recall'], 1 / 3)
        self.assertAlmostEqual(rates['sensitivity_recall_conditional'], .5)
        self.assertAlmostEqual(rates['unscorable_rate'], 1 / 3)
        self.assertTrue(rates['threshold_available'])
        missing = detection_rates(group, None)
        self.assertIsNone(missing['sensitivity_recall'])
        self.assertIsNone(missing['sensitivity_recall_conditional'])
        self.assertFalse(missing['threshold_available'])
        self.assertTrue(missing['unavailable_reason'])

    def test_csv_shards_byte_limit_and_complete_rows(self):
        table = pd.DataFrame(dict(id=range(15), text=['điểm' * 10] * 15))
        single = list(csv_payloads(table, 'objects.csv'))
        self.assertEqual(single[0][0], 'objects.csv')
        shards = list(csv_payloads(table, 'objects.csv', max_bytes=300, shard_rows=4))
        self.assertGreater(len(shards), 1)
        self.assertEqual(shards[0][0], 'objects_000.csv')
        self.assertTrue(all(len(data) < 300 for _, _, data in shards))
        restored = pd.concat([pd.read_csv(StringIO(data.decode('utf-8'))) for _, _, data in shards], ignore_index=True)
        pd.testing.assert_frame_equal(restored, table)
        self.assertEqual(sum(rows for _, rows, _ in shards), len(table))

    def test_overlay_radius_limit_and_selected_object(self):
        obj = KittiObject('Car', 0., 0, 0., np.array([2., 3., 12., 13.]),
                          np.array([2., 2., 4.]), np.array([0., 0., 5.]), 0.)
        frame = dict(image=np.zeros((60, 100, 3), np.uint8), points=np.zeros((12001, 4)),
                     calib=None, labels=[obj])
        uv = np.ones((12001, 2)) * 5
        with patch('src.calibration_benchmark.project', return_value=(uv, np.ones(12001, bool), np.ones(12001))), \
             patch('src.calibration_benchmark.cv2.circle') as circles, \
             patch('src.calibration_benchmark.cv2.rectangle') as rectangles, \
             patch('src.calibration_benchmark.cv2.putText') as texts:
            image = overlay(frame, configurations()[0], 'test', selected_object_id=0)
        self.assertEqual(image.shape, frame['image'].shape)
        self.assertLessEqual(circles.call_count, 6000)
        self.assertGreater(circles.call_count, 0)
        self.assertTrue(all(call.args[2] == 2 for call in circles.call_args_list))
        self.assertEqual(rectangles.call_args_list[0].args[-1], 3)
        self.assertIn('SELECTED #0 Car 5.00m', texts.call_args_list[0].args[1])


if __name__ == '__main__':
    unittest.main()

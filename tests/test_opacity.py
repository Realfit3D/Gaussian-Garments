import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from opacity_controller_v3_common import SplineOpacityController, load_v3, save_v3
from opacity_runtime import GarmentOpacity, add_opacity_arguments


class OpacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(41)
        self.garments = [
            SimpleNamespace(get_opacity=torch.rand(31, 1), get_xyz=torch.randn(31, 3)),
            SimpleNamespace(get_opacity=torch.rand(13, 1), get_xyz=torch.randn(13, 3)),
        ]

    def args(self, mode="original", scale=1, paths=()):
        return SimpleNamespace(opacity_mode=mode, opacity_scale=scale, v3_controller=paths)

    def checkpoint(self, name="shirt", folder="opacity_controller_v3", **changes):
        path = self.root / name / folder / "controller.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_v3(path, SplineOpacityController(), torch.zeros(3), torch.ones(3), 5000, {"subject": name})
        if changes:
            data = torch.load(path, weights_only=True)
            data.update(changes)
            torch.save(data, path)
        return path

    def test_parser_defaults_and_aliases(self):
        parser = argparse.ArgumentParser()
        add_opacity_arguments(parser)
        self.assertEqual(parser.parse_args([]).opacity_mode, "original")
        self.assertEqual(parser.parse_args(["--opacity_scale", ".5"]).opacity_scale, .5)

    def test_original_exact_and_direct_endpoints(self):
        expected = torch.cat([g.get_opacity for g in self.garments])
        for mode in ("original", "direct", "optical_depth"):
            control = GarmentOpacity(self.args(mode), ["a", "b"], self.root, "cpu")
            self.assertTrue(torch.equal(control.apply(self.garments), expected))
        for mode in ("direct", "optical_depth"):
            control = GarmentOpacity(self.args(mode, 0), ["a", "b"], self.root, "cpu")
            self.assertEqual(control.apply(self.garments).count_nonzero(), 0)
            control = GarmentOpacity(self.args(mode, .5), ["a", "b"], self.root, "cpu")
            self.assertTrue(torch.isfinite(control.apply(self.garments)).all())

    def test_invalid_requests(self):
        for mode, scale in (("bad", 1), ("direct", float("nan")), ("direct", float("inf")),
                            ("direct", -.1), ("direct", 1.1), ("original", .5)):
            with self.assertRaises(ValueError):
                GarmentOpacity(self.args(mode, scale), ["a"], self.root, "cpu")

    def test_controller_monotonic_endpoints_and_gradients(self):
        model = SplineOpacityController()
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .1)
        alpha, xyz = self.garments[0].get_opacity, self.garments[0].get_xyz
        outputs = [model(alpha, xyz, s)[0] for s in torch.linspace(0, 1, 41)]
        self.assertTrue(torch.equal(outputs[0], torch.zeros_like(alpha)))
        self.assertTrue(torch.equal(outputs[-1], alpha))
        self.assertTrue(all((a <= b + 1e-7).all() for a, b in zip(outputs, outputs[1:])))
        result, p, _ = model(alpha, xyz, .4)
        torch.testing.assert_close(p.sum(-1), torch.ones(len(alpha)))
        result.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_zero_initialized_controller_matches_optical_depth(self):
        alpha, xyz = self.garments[0].get_opacity, self.garments[0].get_xyz
        actual = SplineOpacityController()(alpha, xyz, .4)[0]
        torch.testing.assert_close(actual, -torch.expm1(.4 * torch.log1p(-alpha)))

    def test_load_and_ordered_multi_garment_runtime(self):
        self.checkpoint("shirt")
        self.checkpoint("pants", "opacity_controller_v3_valid_cameras")
        control = GarmentOpacity(self.args("v3", .4), ["pants", "shirt"], self.root, "cpu")
        control.bind_canonical(self.garments)
        actual = control.apply(self.garments)
        self.assertEqual(tuple(actual.shape), (44, 1))
        expected = torch.cat([SplineOpacityController()(g.get_opacity, g.get_xyz, .4)[0] for g in self.garments])
        torch.testing.assert_close(actual, expected)
        for scale in (0, 1):
            control.scale = scale
            expected = torch.cat([g.get_opacity for g in self.garments]) * scale
            self.assertTrue(torch.equal(control.apply(self.garments), expected))

    def test_checkpoint_validation(self):
        for changes in ({"canonical_scale": torch.zeros(3)},
                        {"canonical_center": torch.zeros(1, 3)},
                        {"canonical_center": torch.full((3,), float("nan"))},
                        {"config": {"subject": "wrong"}}):
            path = self.checkpoint(**changes)
            with self.assertRaises(ValueError):
                load_v3(path, "cpu", subject="shirt")

    def test_v3_coordinates_remain_fixed_during_motion(self):
        path = self.checkpoint()
        data = torch.load(path, weights_only=True)
        data["controller"]["net.4.weight"].normal_(0, .5)
        torch.save(data, path)
        control = GarmentOpacity(self.args("v3", .4), ["shirt"], self.root, "cpu")
        garments = self.garments[:1]
        with self.assertRaises(RuntimeError):
            control.apply(garments)
        control.bind_canonical(garments)
        before = control.apply(garments)
        garments[0].get_xyz.add_(100)
        torch.testing.assert_close(control.apply(garments), before, rtol=0, atol=0)

    def test_targeted_v3_leaves_other_garment_unchanged(self):
        self.checkpoint("shirt")
        args = self.args("v3", .4)
        args.opacity_garments = ["shirt"]
        control = GarmentOpacity(args, ["pants", "shirt"], self.root, "cpu")
        control.bind_canonical(self.garments)
        result = control.apply(self.garments)
        torch.testing.assert_close(result[:31], self.garments[0].get_opacity)
        with self.assertRaises(ValueError):
            args.opacity_garments = ["missing"]
            GarmentOpacity(args, ["pants", "shirt"], self.root, "cpu")

    def test_missing_ambiguous_or_incorrect_mapping(self):
        with self.assertRaises(ValueError):
            GarmentOpacity(self.args("v3"), ["shirt"], self.root, "cpu")
        path = self.checkpoint()
        self.checkpoint(folder="opacity_controller_v3_valid_cameras")
        with self.assertRaises(ValueError):
            GarmentOpacity(self.args("v3"), ["shirt"], self.root, "cpu")
        for paths in (["wrong=x"], ["shirt=x", "shirt=y"], ["shirt"]):
            with self.assertRaises(ValueError):
                GarmentOpacity(self.args("v3", paths=paths), ["shirt"], self.root, "cpu")
        control = GarmentOpacity(self.args("v3", paths=[f"shirt={path}"]), ["shirt"], self.root, "cpu")
        self.assertEqual(len(control.controllers), 1)


if __name__ == "__main__":
    unittest.main()

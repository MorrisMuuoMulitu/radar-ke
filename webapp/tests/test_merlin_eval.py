"""Tests for MERLIN evaluation helpers (pure logic, no torch/GPU)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'RADAR_inference'))
import merlin_eval


class AucTests(unittest.TestCase):
    def test_perfect_inverted_and_tied_separation(self):
        self.assertEqual(merlin_eval.auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(merlin_eval.auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(merlin_eval.auc_score([0, 1], [0.5, 0.5]), 0.5)

    def test_single_class_returns_none(self):
        self.assertIsNone(merlin_eval.auc_score([1, 1, 1], [0.1, 0.2, 0.3]))
        self.assertIsNone(merlin_eval.auc_score([], []))

    def test_matches_known_value_with_a_tie(self):
        # scores 1,2,3,4 with labels 0,1,0,1 -> pairs (1,0)(2,1)(3,0)(4,1): AUC 0.75
        self.assertAlmostEqual(merlin_eval.auc_score([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]), 0.75)


class SelectionTests(unittest.TestCase):
    def test_subset_is_deterministic_and_sized(self):
        ids = [f'AC{i:04d}' for i in range(100)]
        first = merlin_eval.select_subset(ids, n=10, seed=7)
        self.assertEqual(first, merlin_eval.select_subset(ids, n=10, seed=7))
        self.assertEqual(len(first), 10)
        self.assertNotEqual(first, merlin_eval.select_subset(ids, n=10, seed=8))
        self.assertEqual(len(merlin_eval.select_subset(ids, n=500, seed=0)), 100)

    def test_patient_id_strips_suffix(self):
        self.assertEqual(merlin_eval.patient_id('AC1234.nii.gz'), 'AC1234')
        self.assertEqual(merlin_eval.patient_id('AC1234.nii'), 'AC1234')


class MetricsTests(unittest.TestCase):
    def test_compute_aucs_skips_minus_one_and_missing_and_fills_nan(self):
        results = pd.DataFrame({
            'file_name': ['p1.nii.gz', 'p2.nii.gz', 'p3.nii.gz', 'p4.nii.gz'],
            '胆囊_结石': [0.9, 0.1, None, 0.4],
        })
        labels = {'gallstones': {'p1': 1, 'p2': 0, 'p3': 1, 'p4': -1}}
        computed = merlin_eval.compute_aucs(results, labels)
        entry = computed['gallstones']
        self.assertEqual(entry['n'], 3)          # p4 skipped (-1)
        self.assertEqual(entry['positives'], 2)
        self.assertAlmostEqual(entry['auc'], 0.5)  # NaN -> 0 for p3 (positive, low score)

    def test_gallbladder_absence_uses_segmentation_threshold(self):
        results = pd.DataFrame({
            'file_name': ['p1.nii.gz', 'p2.nii.gz'],
            '胆囊_术后胆囊缺失': [1000.0, 999.0],
        })
        labels = {'surgically_absent_gallbladder': {'p1': 1, 'p2': 0}}
        computed = merlin_eval.compute_aucs(results, labels)
        self.assertEqual(computed['surgically_absent_gallbladder']['auc'], 1.0)

    def test_radar_key_strips_parenthesised_english_name(self):
        self.assertEqual(merlin_eval.radar_key('主动脉_主动脉瘤_(abdominal_aortic_aneurysm)'), '主动脉_主动脉瘤')
        self.assertEqual(merlin_eval.radar_key('胆囊_结石'), '胆囊_结石')

    def test_compute_aucs_accepts_inference_column_format(self):
        """Inference writes `中文_(english)` columns; metrics must still match."""
        results = pd.DataFrame({
            'file_name': ['p1.nii.gz', 'p2.nii.gz'],
            '胆囊_术后胆囊缺失_(surgically_absent_gallbladder)': [1000.0, 999.0],
            '胆囊_结石_(gallstones)': [0.8, 0.2],
        })
        labels = {'surgically_absent_gallbladder': {'p1': 1, 'p2': 0},
                  'gallstones': {'p1': 1, 'p2': 0}}
        computed = merlin_eval.compute_aucs(results, labels)
        self.assertEqual(computed['surgically_absent_gallbladder']['auc'], 1.0)
        self.assertEqual(computed['gallstones']['auc'], 1.0)

    def test_compare_to_published_covers_all_findings(self):
        computed = {'gallstones': {'auc': 0.9, 'n': 10, 'positives': 3}}
        rows = merlin_eval.compare_to_published(computed)
        self.assertEqual(len(rows), len(merlin_eval.PUBLISHED_AUCS))
        gallstones = next(r for r in rows if r['disease'] == 'gallstones')
        self.assertAlmostEqual(gallstones['delta'], 0.9 - merlin_eval.PUBLISHED_AUCS['gallstones'])
        missing = next(r for r in rows if r['disease'] == 'appendicitis')
        self.assertIsNone(missing['computed'])
        self.assertIsNone(missing['delta'])

    def test_report_contains_table_and_average(self):
        computed = {'gallstones': {'auc': 0.9, 'n': 4, 'positives': 2}}
        rows = merlin_eval.compare_to_published(computed)
        report = merlin_eval.format_report(rows, computed, n_cases=4, results_csv='results.csv')
        self.assertIn('| finding | published | computed |', report)
        self.assertIn('Average AUC: **0.9000**', report)
        self.assertIn('Cases scored: **4**', report)
        self.assertIn('not a new clinical claim', report)


class DataDirTests(unittest.TestCase):
    def test_list_and_estimate(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('a.nii.gz', 'b.nii.gz', 'notes.txt'):
                (Path(folder) / name).write_bytes(b'x' * 100)
            files = merlin_eval.list_case_files(folder)
            self.assertEqual([f.name for f in files], ['a.nii.gz', 'b.nii.gz'])
            stats = merlin_eval.estimate(folder, n=1)
            self.assertEqual(stats['available'], 2)
            self.assertEqual(stats['selected'], 1)
            self.assertEqual(stats['bytes_selected'], 100)



class PortalIngestionTests(unittest.TestCase):
    """The portal -> JSON -> selection path used by merlin_run.py."""

    def _portal(self, folder):
        import pandas as pd
        portal = Path(folder) / 'merlin_download'
        (portal / 'nifti').mkdir(parents=True)
        pd.DataFrame({
            'study id': ['P1', 'P2', 'P3'],
            'Findings': ['Liver: normal. IMPRESSION: normal.', 'No stones.', 'Spleen enlarged.'],
            'Split': ['test', 'test', 'train'],
            'Few Shot': [0, 0, 0],
        }).to_excel(portal / 'reports_final.xlsx', index=False)
        pd.DataFrame({'id': ['P1', 'P2', 'P3'], 'gallstones': [1, 0, -1]}).to_csv(
            portal / 'zero_shot_findings_disease_cls.csv', index=False)
        for pid in ('P1', 'P2'):
            (portal / 'nifti' / f'{pid}.nii.gz').write_bytes(b'x' * 10)
        return portal

    @unittest.skipUnless(__import__('importlib').util.find_spec('openpyxl'), 'openpyxl not installed')
    def test_build_json_inputs_from_portal_files(self):
        with tempfile.TemporaryDirectory() as folder:
            portal = self._portal(folder)
            report = merlin_eval.build_report_json(portal / 'reports_final.xlsx', Path(folder) / 'report.json')
            labels = merlin_eval.build_labels_json(portal / 'zero_shot_findings_disease_cls.csv',
                                                   Path(folder) / 'labels.json')
            self.assertEqual(report['P1']['split'], 'test')
            self.assertIn('findings', report['P1'])
            self.assertTrue(report['P1']['report'].endswith('normal.'))
            self.assertEqual(labels['gallstones'], {'P1': 1, 'P2': 0, 'P3': -1})
            self.assertTrue((Path(folder) / 'report.json').exists())

    @unittest.skipUnless(__import__('importlib').util.find_spec('openpyxl'), 'openpyxl not installed')
    def test_subset_is_selected_by_file_name(self):
        """Regression: staging/resume compare file names, not patient ids."""
        import argparse
        import importlib
        with tempfile.TemporaryDirectory() as folder:
            portal = self._portal(folder)
            ckpt = Path(folder) / 'ckpt'
            ckpt.mkdir()
            os.environ['CONFIGS_ROOT'] = str(ckpt)
            os.environ['MODEL_ROOT'] = str(ckpt)
            try:
                import merlin_run
                importlib.reload(merlin_run)
                args = argparse.Namespace(portal_dir=str(portal), out=str(Path(folder) / 'out'), n=2,
                                          all=False, seed=0, chunk=1, results=None, save_tag='test',
                                          check=True, dry_run=False, rebuild_json=False, delete_source=False)
                plan = merlin_run.readiness(args)
                self.assertIsNotNone(plan)
                self.assertEqual(plan['selected'], ['P1.nii.gz', 'P2.nii.gz'])
                self.assertEqual(sorted(plan['paths_by_name']), ['P1.nii.gz', 'P2.nii.gz'])
                self.assertNotIn('P3.nii.gz', plan['selected'])       # train split excluded
            finally:
                os.environ.pop('CONFIGS_ROOT', None)
                os.environ.pop('MODEL_ROOT', None)


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()

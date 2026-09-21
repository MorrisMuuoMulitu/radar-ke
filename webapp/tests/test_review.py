import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review

class ReviewTests(unittest.TestCase):
    def test_literal_search_and_missing_scores(self):
        rows = review.score_table({'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': None})
        self.assertEqual(len(review.filter_findings(rows, '[', [], 0, False)), 0)
        self.assertEqual(len(review.filter_findings(rows, '', [], 0.5, True)), 1)
        self.assertEqual(rows.iloc[0]['finding'], 'Cyst')
        self.assertEqual(rows.iloc[0]['organ'], 'Liver')

    def test_clear_removes_owned_files_and_state(self):
        with tempfile.TemporaryDirectory() as root:
            work = Path(root) / 'radar_web_test'
            work.mkdir()
            (work / 'scan.nii').write_text('test')
            state = {'work_dir': str(work), 'result': {}, 'review_notes': 'old', 'unrelated': 1}
            review.clear_case(state)
            self.assertFalse(work.exists())
            self.assertNotIn('result', state)
            self.assertNotIn('review_notes', state)
            self.assertEqual(state['unrelated'], 1)

    def test_missing_predictions_are_not_exported_as_zero(self):
        rows = review.score_table({'Liver_Cyst': None})
        self.assertTrue(rows['score'].isna().all())
        self.assertEqual(len(review.filter_findings(rows, '', [], 0, True)), 0)

    def test_case_history_round_trip_saves_review_without_volume_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            record = review.build_case_record(
                {'file_name': 'case.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.8}, 'case_path': '/tmp/private.npz'},
                status='Reviewed',
                shortlist=['原文 (Liver_Cyst)'],
                notes='Correlate with prior imaging.',
            )
            saved = review.save_case_record(record, Path(folder))
            loaded = review.load_case_record(saved['case_id'], Path(folder))
            cases = review.list_case_records(Path(folder))
            self.assertEqual(loaded['file_name'], 'case.nii.gz')
            self.assertEqual(loaded['status'], 'Reviewed')
            self.assertNotIn('case_path', loaded)
            self.assertNotIn('image', loaded)
            self.assertEqual(cases[0]['case_id'], saved['case_id'])

    def test_structured_report_includes_shortlist_scores_notes_and_notice(self):
        record = review.build_case_record(
            {'file_name': 'case.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': 0.2}},
            status='Reviewed',
            shortlist=['原文 (Liver_Cyst)'],
            notes='Simple cyst favored.',
        )
        report = review.structured_report(record)
        self.assertIn('Case: case.nii.gz', report)
        self.assertIn('Review status: Reviewed', report)
        self.assertIn('Liver / Cyst: 0.800', report)
        self.assertIn('Simple cyst favored.', report)
        self.assertIn('qualified radiologist review', report)

    def test_structured_report_uses_confirmed_findings_before_shortlist(self):
        record = review.build_case_record(
            {'file_name': 'case.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': 0.2}},
            status='Reviewed',
            shortlist=['原文 (Kidney_Cyst)'],
            notes='Prior available.',
            finding_states={'原文 (Liver_Cyst)': 'Likely present', '原文 (Kidney_Cyst)': 'Likely absent'},
        )
        report = review.structured_report(record)
        self.assertIn('Confirmed findings:', report)
        self.assertIn('Liver / Cyst: 0.800', report)
        self.assertNotIn('Kidney / Cyst: 0.200', report)
        self.assertIn('Likely present: 1', report)

    def test_window_presets_map_hu_to_display_range(self):
        image = review.apply_window([-1000, 40, 400], center=40, width=400)
        self.assertEqual(float(image[0]), 0.0)
        self.assertAlmostEqual(float(image[1]), 0.5, places=2)
        self.assertEqual(float(image[2]), 1.0)
        self.assertIn('Abdomen', review.WINDOW_PRESETS)

if __name__ == '__main__':
    unittest.main()

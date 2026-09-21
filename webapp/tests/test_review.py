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

if __name__ == '__main__':
    unittest.main()

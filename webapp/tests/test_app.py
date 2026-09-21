"""Exercise the review workflow with a small synthetic model-output volume."""
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'

class WorkspaceTests(unittest.TestCase):
    def test_review_filters_overlay_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='radar_web_') as folder:
            case = Path(folder) / 'case.npz'
            image = np.ones((10, 12, 14), dtype=np.float32) * 0.5
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask[2:6, 3:7, 4:8] = 21
            np.savez(case, image=image, mask=mask)
            app = AppTest.from_file(str(APP), default_timeout=30)
            app.session_state['work_dir'] = folder
            app.session_state['result'] = {'file_name': 'synthetic.nii.gz', 'case_path': str(case),
                'scores': {'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': None}}
            app.run()
            self.assertFalse(app.exception)
            next(s for s in app.selectbox if s.label == 'Window preset').set_value('Bone').run()
            axial_slider = next(s for s in app.slider if s.label == 'Axial slice')
            self.assertEqual(axial_slider.value, 5)
            next(b for b in app.button if b.label == 'Next axial slice').click().run()
            self.assertEqual(app.session_state['slice_0'], 6)
            next(b for b in app.button if b.label == 'Previous axial slice').click().run()
            self.assertEqual(app.session_state['slice_0'], 5)
            next(s for s in app.selectbox if s.label == 'Highlight anatomy').set_value(21).run()
            next(b for b in app.button if b.label == 'Center on selected anatomy').click().run()
            self.assertEqual(app.session_state['slice_0'], 3)
            next(t for t in app.text_input if t.label == 'Search findings').set_value('[').run()
            self.assertTrue(any('No findings match' in i.value for i in app.info))
            next(t for t in app.text_area if t.label == 'Reviewer notes').set_value('Test note').run()
            next(s for s in app.multiselect if s.label == 'Shortlist findings for follow-up').set_value(['原文 (Liver_Cyst)']).run()
            next(s for s in app.selectbox if s.label == 'Finding to mark').set_value('原文 (Liver_Cyst)').run()
            next(s for s in app.selectbox if s.label == 'Finding state').set_value('Likely present').run()
            next(b for b in app.button if b.label == 'Apply finding state').click().run()
            self.assertEqual(app.session_state['finding_states']['原文 (Liver_Cyst)'], 'Likely present')
            self.assertEqual(app.session_state['review_notes'], 'Test note')
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == 'Clear case and delete uploads').click().run()
            self.assertFalse(case.exists())
            self.assertNotIn('result', app.session_state)
            self.assertFalse(app.exception)

    def test_review_can_save_history_and_open_saved_report(self):
        with tempfile.TemporaryDirectory() as history:
            os.environ['RADAR_CASE_HISTORY_DIR'] = history
            try:
                app = AppTest.from_file(str(APP), default_timeout=30)
                app.session_state['result'] = {'file_name': 'history_case.nii.gz', 'example': True,
                    'image': np.ones((4, 5, 6), dtype=np.float32) * 0.5, 'mask': None,
                    'scores': {'原文 (Liver_Cyst)': 0.8}}
                app.run()
                next(t for t in app.text_area if t.label == 'Reviewer notes').set_value('Follow up with prior study.').run()
                next(s for s in app.multiselect if s.label == 'Shortlist findings for follow-up').set_value(['原文 (Liver_Cyst)']).run()
                next(s for s in app.selectbox if s.label == 'Finding to mark').set_value('原文 (Liver_Cyst)').run()
                next(s for s in app.selectbox if s.label == 'Finding state').set_value('Likely present').run()
                next(b for b in app.button if b.label == 'Apply finding state').click().run()
                next(b for b in app.button if b.label == 'Save case to history').click().run()
                self.assertTrue(list(Path(history).glob('*.json')))
                app.run()
                self.assertTrue(any('history_case.nii.gz' in option for s in app.selectbox if s.label == 'Saved cases' for option in s.options))
                next(b for b in app.button if b.label == 'Open saved review').click().run()
                self.assertTrue(app.session_state['result']['history_only'])
                self.assertTrue(any('Structured report draft' in h.value for h in app.subheader))
                self.assertFalse(app.exception)
            finally:
                os.environ.pop('RADAR_CASE_HISTORY_DIR', None)

if __name__ == '__main__':
    unittest.main()

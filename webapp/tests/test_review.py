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
        self.assertIn('Study: case.nii.gz', report)
        self.assertIn('Review status: Reviewed', report)
        self.assertIn('- Cyst: 0.800', report)
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
        self.assertIn('FINDINGS BY ORGAN', report)
        self.assertIn('Liver:', report)
        self.assertIn('- Cyst: 0.800', report)
        self.assertNotIn('Kidney:', report)
        self.assertIn('Likely present: 1', report)

    def test_structured_report_groups_confirmed_findings_by_anatomy(self):
        record = review.build_case_record(
            {'file_name': 'case.nii.gz',
             'scores': {'原文 (Liver_Cyst)': 0.8, '原文 (Liver_Abscess)': 0.7, '原文 (Kidney_Cyst)': 0.2}},
            status='Reviewed',
            shortlist=[],
            notes='',
            finding_states={'原文 (Liver_Cyst)': 'Likely present', '原文 (Liver_Abscess)': 'Likely present'},
        )
        report = review.structured_report(record)
        self.assertIn('Liver:', report)
        self.assertIn('- Cyst: 0.800', report)
        self.assertIn('- Abscess: 0.700', report)

    def test_structured_report_has_sections_and_clinical_reviewer_context(self):
        record = review.build_case_record(
            {'file_name': 'case.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.8}},
            status='Reviewed',
            shortlist=[],
            notes='',
            finding_states={'原文 (Liver_Cyst)': 'Likely present'},
            clinical_context='RUQ pain, suspected cholecystitis',
            reviewer='Dr. Test',
        )
        report = review.structured_report(record)
        for section in ('CLINICAL CONTEXT', 'FINDINGS BY ORGAN', 'IMPRESSION', 'REVIEW LIMITATIONS'):
            self.assertIn(section, report)
        self.assertIn('RUQ pain, suspected cholecystitis', report)
        self.assertIn('Dr. Test', report)
        self.assertIn('Findings marked likely present:', report)
        self.assertIn('Liver \u2014 Cyst: 0.800', report)

    def test_structured_report_empty_case_shows_no_significant_findings(self):
        record = review.build_case_record(
            {'file_name': 'empty.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.1}},
            status='Not started',
            shortlist=[],
            notes='',
        )
        report = review.structured_report(record)
        self.assertIn('No significant findings', report)
        self.assertIn('No definite abnormalities flagged', report)
        self.assertIn('Not provided.', report)

    def test_structured_report_excludes_absent_and_ignored_findings(self):
        record = review.build_case_record(
            {'file_name': 'case.nii.gz',
             'scores': {'原文 (Liver_Cyst)': 0.9, '原文 (Liver_Abscess)': 0.8, '原文 (Kidney_Cyst)': 0.7}},
            status='In progress',
            shortlist=[],
            notes='',
            finding_states={'原文 (Liver_Cyst)': 'Likely absent', '原文 (Liver_Abscess)': 'Ignore',
                            '原文 (Kidney_Cyst)': 'Likely present'},
        )
        report = review.structured_report(record)
        self.assertNotIn('Liver:', report)
        self.assertIn('Kidney:', report)
        self.assertIn('- Cyst: 0.700', report)

    def test_window_presets_map_hu_to_display_range(self):
        image = review.apply_window([-1000, 40, 400], center=40, width=400)
        self.assertEqual(float(image[0]), 0.0)
        self.assertAlmostEqual(float(image[1]), 0.5, places=2)
        self.assertEqual(float(image[2]), 1.0)
        self.assertIn('Abdomen', review.WINDOW_PRESETS)

    def test_worklist_rows_reports_likely_present_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            record = review.build_case_record(
                {'file_name': 'case.nii.gz',
                 'scores': {'原文 (Liver_Cyst)': 0.8, '原文 (Liver_Abscess)': 0.7, '原文 (Kidney_Cyst)': 0.2}},
                status='In progress',
                shortlist=['原文 (Liver_Cyst)'],
                notes='',
                finding_states={'原文 (Liver_Cyst)': 'Likely present', '原文 (Liver_Abscess)': 'Likely present',
                                '原文 (Kidney_Cyst)': 'Likely absent'},
            )
            review.save_case_record(record, Path(folder))
            rows = review.worklist_rows(Path(folder))
            self.assertEqual(len(rows), 1)
            row = rows.iloc[0]
            self.assertEqual(row['file_name'], 'case.nii.gz')
            self.assertEqual(row['status'], 'In progress')
            self.assertEqual(row['shortlist_count'], 1)
            self.assertEqual(row['likely_present_count'], 2)
            self.assertIn('Liver / Cyst', row['likely_present'])
            self.assertIn('Liver / Abscess', row['likely_present'])
            self.assertNotIn('Kidney', row['likely_present'])

    def test_filter_worklist_searches_and_filters_by_status(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, status in [('alpha.nii.gz', 'Reviewed'), ('beta.nii.gz', 'Not started')]:
                record = review.build_case_record({'file_name': name, 'scores': {}}, status, [], '')
                review.save_case_record(record, Path(folder))
            rows = review.worklist_rows(Path(folder))
            self.assertEqual(len(review.filter_worklist(rows, query='beta')), 1)
            self.assertEqual(review.filter_worklist(rows, query='beta').iloc[0]['file_name'], 'beta.nii.gz')
            self.assertEqual(len(review.filter_worklist(rows, statuses=['Reviewed'])), 1)
            self.assertEqual(len(review.filter_worklist(rows, query='nope', statuses=['Reviewed'])), 0)
            self.assertEqual(len(review.filter_worklist(rows, statuses=[])), 2)

    def test_delete_case_record_removes_json_only(self):
        with tempfile.TemporaryDirectory() as folder:
            record = review.build_case_record({'file_name': 'case.nii.gz', 'scores': {}}, 'Not started', [], '')
            saved = review.save_case_record(record, Path(folder))
            self.assertTrue(Path(folder, saved['case_id'] + '.json').exists())
            self.assertTrue(review.delete_case_record(saved['case_id'], Path(folder)))
            self.assertFalse(Path(folder, saved['case_id'] + '.json').exists())
            self.assertFalse(review.delete_case_record(saved['case_id'], Path(folder)))

    def test_match_findings_to_report_suggests_candidates(self):
        scores = {'原文 (Liver_Cyst)': 0.8, '原文 (Gallbladder_Cholecystolithiasis)': 0.7,
                  '原文 (Kidney_Cyst)': 0.4}
        matched = review.match_findings_to_report('Liver cyst and gallstones noted.', scores)
        self.assertIn('原文 (Liver_Cyst)', matched)
        self.assertIn('原文 (Gallbladder_Cholecystolithiasis)', matched)
        self.assertNotIn('原文 (Kidney_Cyst)', matched)
        self.assertEqual(review.match_findings_to_report('Normal study.', scores), [])

    def test_validation_table_and_summary_compute_confusion_matrix(self):
        scores = {'原文 (Liver_Cyst)': 0.8, '原文 (Pancreas_Cyst)': 0.3,
                  '原文 (Spleen_Cyst)': 0.9, '原文 (Kidney_Cyst)': 0.2}
        record = {'scores': scores, 'validation': {
            'present': ['原文 (Liver_Cyst)', '原文 (Pancreas_Cyst)'],
            'absent': ['原文 (Spleen_Cyst)', '原文 (Kidney_Cyst)']}}
        table = review.validation_table(record, threshold=0.5)
        summary = review.validation_summary(record, threshold=0.5)
        self.assertEqual(dict(table.set_index('key')['agreement']), {
            '原文 (Liver_Cyst)': 'TP', '原文 (Pancreas_Cyst)': 'FN',
            '原文 (Spleen_Cyst)': 'FP', '原文 (Kidney_Cyst)': 'TN'})
        self.assertEqual(summary['n'], 4)
        self.assertEqual(summary['tp'], 1)
        self.assertEqual(summary['fp'], 1)
        self.assertEqual(summary['tn'], 1)
        self.assertEqual(summary['fn'], 1)
        self.assertAlmostEqual(summary['accuracy'], 0.5)
        self.assertAlmostEqual(summary['sensitivity'], 0.5)
        self.assertAlmostEqual(summary['specificity'], 0.5)
        self.assertAlmostEqual(summary['precision'], 0.5)
        self.assertAlmostEqual(summary['f1'], 0.5)

    def test_validation_skips_unknown_keys_and_missing_scores(self):
        scores = {'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': None}
        record = {'scores': scores, 'validation': {
            'present': ['原文 (Kidney_Cyst)', 'not_a_real_finding'],
            'absent': []}}
        summary = review.validation_summary(record, threshold=0.5)
        self.assertEqual(summary['n'], 1)
        self.assertEqual(summary['fn'], 1)  # present but score missing -> not predicted

    def test_validation_report_contains_metrics_and_rows(self):
        scores = {'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': 0.2}
        record = {'file_name': 'case.nii.gz', 'scores': scores, 'validation': {
            'present': ['原文 (Liver_Cyst)'], 'absent': ['原文 (Kidney_Cyst)']}}
        text = review.validation_report(record, threshold=0.5)
        self.assertIn('VALIDATION SUMMARY', text)
        self.assertIn('Accuracy:', text)
        self.assertIn('Liver / Cyst', text)
        self.assertIn('TP', text)
        self.assertIn('TN', text)

    def test_build_case_record_persists_validation(self):
        record = review.build_case_record(
            {'file_name': 'case.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.8}},
            'Reviewed', [], '',
            validation={'present': ['原文 (Liver_Cyst)'], 'absent': [], 'threshold': 0.6})
        self.assertEqual(record['validation']['present'], ['原文 (Liver_Cyst)'])
        self.assertEqual(record['validation']['threshold'], 0.6)

    def test_extract_reference_splits_negated_and_clause_attributed_findings(self):
        """Real report phrasings: negation must not leak across organs."""
        scores = {'原文 (Small bowel_Obstruction)': 0.4, '原文 (Large bowel_Obstruction)': 0.3,
                  '原文 (Kidney_Hydronephrosis)': 0.2}
        text = ('Small Bowel: consistent with a mechanical small-bowel obstruction. '
                'Large Bowel: No focal large-bowel obstruction identified. '
                'Kidneys: preserved morphology without hydronephrosis.')
        ref = review.extract_reference(text, scores)
        self.assertIn('原文 (Small bowel_Obstruction)', ref['present'])
        self.assertIn('原文 (Large bowel_Obstruction)', ref['absent'])
        self.assertIn('原文 (Kidney_Hydronephrosis)', ref['absent'])
        self.assertNotIn('原文 (Large bowel_Obstruction)', ref['present'])
        # present-only helper must exclude negated findings
        present_only = review.match_findings_to_report(text, scores)
        self.assertEqual(present_only, ['原文 (Small bowel_Obstruction)'])

    def test_extract_reference_uses_word_boundaries(self):
        """'gallbladder'/'gallstones' must not trigger bladder findings."""
        scores = {'原文 (Bladder_Stone)': 0.2, '原文 (Gallbladder_Cholecystolithiasis)': 0.9,
                  '原文 (Kidney_Nephrolithiasis)': 0.3}
        ref = review.extract_reference('Gallbladder: No calcified gallstones.', scores)
        self.assertIn('原文 (Gallbladder_Cholecystolithiasis)', ref['absent'])
        self.assertNotIn('原文 (Bladder_Stone)', ref['present'] + ref['absent'])

    def test_extract_reference_maps_common_report_synonyms(self):
        scores = {'原文 (Liver_Hypoattenuating lesion)': 0.8, '原文 (Spleen_Splenomegaly)': 0.6,
                  '原文 (Large bowel_Colon cancer)': 0.7, '原文 (Kidney_Hypoattenuating lesion)': 0.5}
        text = ('Liver: Multiple hypodense lesions are seen diffusely involving both hepatic lobes. '
                'Spleen: Enlarged in size, measuring 122 x 66 x 132 mm. '
                'Colon: mural thickening of the splenic flexure, suspicious for a colonic neoplastic lesion.')
        ref = review.extract_reference(text, scores)
        self.assertIn('原文 (Liver_Hypoattenuating lesion)', ref['present'])
        self.assertIn('原文 (Spleen_Splenomegaly)', ref['present'])
        self.assertIn('原文 (Large bowel_Colon cancer)', ref['present'])
        self.assertNotIn('原文 (Kidney_Hypoattenuating lesion)', ref['present'])

    def test_extract_reference_ignores_differential_and_recommendation_clauses(self):
        scores = {'原文 (Liver_Abscess)': 0.5, '原文 (Liver_Hypoattenuating lesion)': 0.8}
        text = ('Liver: multiple hypodense lesions are seen diffusely involving both hepatic lobes. '
                'Differential: infective microabscesses; multifocal hepatic infarcts. '
                'Recommendation: correlation with MRI liver.')
        ref = review.extract_reference(text, scores)
        self.assertIn('原文 (Liver_Hypoattenuating lesion)', ref['present'])
        self.assertNotIn('原文 (Liver_Abscess)', ref['present'] + ref['absent'])

    def test_cohort_validation_rolls_up_across_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            # case A: Liver_Cyst present+predicted positive (TP); Kidney_Cyst absent but predicted (FP)
            review.save_case_record(review.build_case_record(
                {'file_name': 'a.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.8, '原文 (Kidney_Cyst)': 0.7}},
                'Reviewed', [], '', validation={'present': ['原文 (Liver_Cyst)'],
                                                'absent': ['原文 (Kidney_Cyst)'], 'threshold': 0.5}), Path(folder))
            # case B: Liver_Cyst present but scored low (FN); Kidney_Cyst absent, scored low (TN)
            review.save_case_record(review.build_case_record(
                {'file_name': 'b.nii.gz', 'scores': {'原文 (Liver_Cyst)': 0.2, '原文 (Kidney_Cyst)': 0.1}},
                'Reviewed', [], '', validation={'present': ['原文 (Liver_Cyst)'],
                                                'absent': ['原文 (Kidney_Cyst)'], 'threshold': 0.5}), Path(folder))
            rows = review.cohort_validation_rows(Path(folder))
            by_key = {row['key']: row for row in rows.to_dict('records')}
            liver = by_key['原文 (Liver_Cyst)']
            kidney = by_key['原文 (Kidney_Cyst)']
            self.assertEqual((liver['cases'], liver['tp'], liver['fn']), (2, 1, 1))
            self.assertAlmostEqual(liver['sensitivity'], 0.5)
            self.assertEqual((kidney['cases'], kidney['fp'], kidney['tn']), (2, 1, 1))
            self.assertAlmostEqual(kidney['specificity'], 0.5)
            summary = review.cohort_validation_summary(rows)
            self.assertEqual((summary['tp'], summary['fp'], summary['tn'], summary['fn']), (1, 1, 1, 1))
            self.assertAlmostEqual(summary['accuracy'], 0.5)
            self.assertIn('COHORT VALIDATION SUMMARY', review.cohort_validation_report(rows))

    def test_cohort_validation_empty_store(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = review.cohort_validation_rows(Path(folder))
            self.assertTrue(rows.empty)
            summary = review.cohort_validation_summary(rows)
            self.assertEqual(summary['findings'], 0)
            self.assertIsNone(summary['sensitivity'])
            self.assertIn('No cases with adjudicated findings', review.cohort_validation_report(rows))

if __name__ == '__main__':
    unittest.main()

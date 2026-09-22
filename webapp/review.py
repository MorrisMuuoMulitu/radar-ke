"""Data operations for the CT review workspace, independent of Streamlit."""
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

FINDING_STATES = ['Needs review', 'Likely present', 'Likely absent', 'Ignore']

REVIEW_STATUSES = ['Not started', 'In progress', 'Reviewed']

# Radiology-style anatomy ordering for the Findings by organ report section.
REPORT_ORGAN_ORDER = [
    'Liver', 'Gallbladder', 'Pancreas', 'Spleen', 'Kidney', 'Adrenal gland',
    'Stomach', 'Duodenum', 'Large bowel', 'Small bowel', 'Esophagus',
    'Aorta', 'Portal vein', 'Heart', 'Lung', 'Bladder', 'Rib', 'Sacrum',
]

# Unmarked findings at/above this model score are still surfaced in the report.
REPORT_SCORE_THRESHOLD = 0.5

WINDOW_PRESETS = {
    'Abdomen': {'center': 60, 'width': 400},
    'Liver': {'center': 70, 'width': 150},
    'Soft tissue': {'center': 40, 'width': 350},
    'Lung': {'center': -600, 'width': 1500},
    'Bone': {'center': 300, 'width': 1500},
}


def apply_window(image, center, width):
    values = np.asarray(image, dtype=np.float32)
    lower = center - width / 2
    upper = center + width / 2
    return np.clip((values - lower) / max(upper - lower, 1), 0, 1)


def score_table(scores):
    rows = []
    for key, value in scores.items():
        label = key.split(' (', 1)[1].removesuffix(')') if ' (' in key else key
        organ, _, finding = label.partition('_')
        rows.append({'key': key, 'organ': organ, 'finding': finding or label, 'score': value})
    return pd.DataFrame(rows, columns=['key', 'organ', 'finding', 'score']).astype({'score': float}).sort_values('score', ascending=False, na_position='last')


def filter_findings(rows, query, organs, threshold, above_only):
    selected = rows.copy()
    if query.strip():
        selected = selected[selected['key'].str.contains(query.strip(), case=False, regex=False, na=False)]
    if organs:
        selected = selected[selected['organ'].isin(organs)]
    if above_only:
        selected = selected[selected['score'] >= threshold]
    return selected


def clear_case(state):
    work = state.get('work_dir')
    if work and Path(work).name.startswith('radar_web_'):
        shutil.rmtree(work, ignore_errors=True)
    for key in list(state):
        if key in ('work_dir', 'result', 'review_notes', 'review_flags', 'review_status',
                   'finding_states', 'review_context',
                   'validation_present', 'validation_absent', 'validation_threshold',
                   'validation_report_text', '_extract_keys') or key.startswith(('slice_', 'selected_', 'pending_')):
            state.pop(key, None)


def history_dir():
    return Path(os.environ.get('RADAR_CASE_HISTORY_DIR', Path.home() / '.radar_ke' / 'cases'))


def _safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def _case_id(file_name, saved_at):
    seed = f'{file_name}|{saved_at}'.encode('utf-8')
    return hashlib.sha256(seed).hexdigest()[:16]


def build_case_record(result, status, shortlist, notes, finding_states=None, saved_at=None,
                      clinical_context=None, reviewer=None, validation=None):
    saved_at = saved_at or datetime.now(timezone.utc).isoformat()
    scores = {key: _safe_float(value) for key, value in result.get('scores', {}).items()}
    file_name = result.get('file_name', 'unknown_case')
    return {
        'case_id': _case_id(file_name, saved_at),
        'file_name': file_name,
        'saved_at': saved_at,
        'status': status,
        'shortlist': list(shortlist or []),
        'finding_states': {key: value for key, value in (finding_states or {}).items() if value in FINDING_STATES},
        'notes': notes or '',
        'clinical_context': clinical_context or '',
        'reviewer': reviewer or '',
        'validation': {
            'present': list((validation or {}).get('present', [])),
            'absent': list((validation or {}).get('absent', [])),
            'threshold': float((validation or {}).get('threshold', REPORT_SCORE_THRESHOLD)),
        },
        'scores': scores,
        'example': bool(result.get('example', False)),
        'notice': 'Research use only. Model scores are not calibrated disease probabilities.',
    }


def save_case_record(record, store_dir=None):
    store = Path(store_dir) if store_dir is not None else history_dir()
    store.mkdir(parents=True, exist_ok=True)
    path = store / f'{record["case_id"]}.json'
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    return record


def load_case_record(case_id, store_dir=None):
    store = Path(store_dir) if store_dir is not None else history_dir()
    path = store / f'{case_id}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def list_case_records(store_dir=None):
    store = Path(store_dir) if store_dir is not None else history_dir()
    if not store.exists():
        return []
    records = []
    for path in sorted(store.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        records.append({
            'case_id': record.get('case_id', path.stem),
            'file_name': record.get('file_name', 'unknown_case'),
            'saved_at': record.get('saved_at', ''),
            'status': record.get('status', 'Not started'),
            'shortlist_count': len(record.get('shortlist', [])),
        })
    return records


def _finding_labels(scores):
    rows = score_table(scores)
    return dict(zip(rows.key, rows.organ + ' / ' + rows.finding))


WORKLIST_COLUMNS = ['case_id', 'file_name', 'saved_at', 'status',
                    'shortlist_count', 'likely_present_count', 'likely_present']


def worklist_rows(store_dir=None):
    """Enriched worklist rows for the case dashboard.

    Adds the likely-present finding count/names (from finding_states) on top
    of the summary fields in list_case_records.
    """
    store = Path(store_dir) if store_dir is not None else history_dir()
    if not store.exists():
        return pd.DataFrame(columns=WORKLIST_COLUMNS)
    records = []
    for path in sorted(store.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        labels = _finding_labels(record.get('scores', {}))
        finding_states = record.get('finding_states', {})
        likely = [labels.get(key, key)
                  for key, state in finding_states.items() if state == 'Likely present']
        records.append({
            'case_id': record.get('case_id', path.stem),
            'file_name': record.get('file_name', 'unknown_case'),
            'saved_at': record.get('saved_at', ''),
            'status': record.get('status', 'Not started'),
            'shortlist_count': len(record.get('shortlist', [])),
            'likely_present_count': len(likely),
            'likely_present': ', '.join(likely),
        })
    return pd.DataFrame(records, columns=WORKLIST_COLUMNS)


def filter_worklist(rows, query='', statuses=None, sort='Newest'):
    """Filter/sort worklist rows (kept here so the UI stays thin and testable)."""
    selected = rows.copy()
    if query.strip():
        selected = selected[selected['file_name'].str.contains(query.strip(), case=False, regex=False, na=False)]
    if statuses:
        selected = selected[selected['status'].isin(statuses)]
    if sort == 'Oldest':
        selected = selected.sort_values('saved_at')
    elif sort == 'Filename':
        selected = selected.sort_values('file_name')
    elif sort == 'Status':
        selected = selected.sort_values(['status', 'saved_at'], ascending=[True, False])
    else:  # Newest
        selected = selected.sort_values('saved_at', ascending=False)
    return selected.reset_index(drop=True)


def delete_case_record(case_id, store_dir=None):
    """Remove a saved case record. Returns True if a file was removed."""
    store = Path(store_dir) if store_dir is not None else history_dir()
    path = store / f'{case_id}.json'
    if not path.exists():
        return False
    path.unlink()
    return True


def _organ_sort_key(organ):
    try:
        return (0, REPORT_ORGAN_ORDER.index(organ))
    except ValueError:
        return (1, organ)


def _flagged_by_organ(record):
    """Group report-worthy findings by anatomy (radiology order).

    Includes findings the reviewer flagged (Likely present / Needs review),
    shortlisted findings, and unmarked findings at or above the score
    threshold. Findings marked Likely absent or Ignore are excluded.
    """
    finding_states = record.get('finding_states', {})
    shortlist = set(record.get('shortlist', []))
    rows = score_table(record.get('scores', {}))
    grouped = {}
    for _, row in rows.iterrows():
        key = row['key']
        state = finding_states.get(key)
        if state in ('Likely absent', 'Ignore'):
            continue
        score = row['score']
        if state in ('Likely present', 'Needs review') or key in shortlist or (
                score is not None and score >= REPORT_SCORE_THRESHOLD):
            if state == 'Likely present':
                display_state = 'Likely present'
            elif state == 'Needs review':
                display_state = 'Needs review'
            elif key in shortlist:
                display_state = 'Shortlisted'
            else:
                display_state = 'High model score'
            grouped.setdefault(row['organ'], []).append((row['finding'], score, display_state))
    return grouped


def _score_text(score):
    if score is None or pd.isna(score):
        return 'unavailable'
    return f'{float(score):.3f}'


def structured_report(record):
    """Radiology-style draft report with Clinical context, Findings by organ,
    Impression, and Review limitations sections."""
    finding_states = record.get('finding_states', {})
    state_counts = {state: list(finding_states.values()).count(state) for state in FINDING_STATES}
    shortlist = list(record.get('shortlist', []))
    rows = score_table(record.get('scores', {}))
    labels = dict(zip(rows.key, rows.organ + ' / ' + rows.finding))
    grouped = _flagged_by_organ(record)
    confirmed = [(organ, finding, score)
                 for organ, items in grouped.items()
                 for finding, score, state in items if state == 'Likely present']
    shortlist_display = [key for key in shortlist
                         if finding_states.get(key) not in ('Likely absent', 'Ignore')]

    lines = [
        '=' * 62,
        ' RADAR-AIDED ABDOMINAL CT REPORT (DRAFT)',
        '=' * 62,
        f'Study: {record.get("file_name", "unknown_case")}',
        f'Reviewed on: {str(record.get("saved_at", ""))[:19].replace("T", " ")}',
        f'Review status: {record.get("status", "Not started")}',
        f'Reviewer: {record.get("reviewer", "") or "—"}',
        '',
        'CLINICAL CONTEXT',
        '----------------',
        f'Indication: {record.get("clinical_context", "") or "Not provided."}',
        'Analysis method: contrast-enhanced abdominal CT interpreted with the RADAR',
        'research model using 146 predefined finding scores.',
        '',
        'FINDINGS BY ORGAN',
        '-----------------',
    ]
    if not grouped:
        lines.append('No significant findings identified in the model analysis.')
    else:
        for organ in sorted(grouped.keys(), key=_organ_sort_key):
            lines.append(f'{organ}:')
            for finding, score, state in grouped[organ]:
                lines.append(f'  - {finding}: {_score_text(score)} ({state})')

    lines.extend([
        '',
        'IMPRESSION',
        '----------',
    ])
    if confirmed:
        lines.append('Findings marked likely present:')
        for organ, finding, score in confirmed:
            lines.append(f'  - {organ} \u2014 {finding}: {_score_text(score)}')
    if shortlist_display:
        lines.append('Shortlisted for follow-up:')
        for key in shortlist_display:
            lines.append(f'  - {labels.get(key, key)}')
    if not confirmed and not shortlist_display:
        lines.append('No definite abnormalities flagged in this analysis.')
    lines.append('Draft impression \u2014 each item requires radiologist confirmation before signing.')

    lines.extend([
        '',
        'REVIEW SUMMARY',
        '--------------',
        f'Likely present: {state_counts["Likely present"]} \u2022 '
        f'Likely absent: {state_counts["Likely absent"]} \u2022 '
        f'Needs review: {state_counts["Needs review"]} \u2022 '
        f'Ignored: {state_counts["Ignore"]} \u2022 Shortlisted: {len(shortlist_display)}',
        '',
        'REVIEWER NOTES',
        '--------------',
        record.get('notes', '').strip() or 'No notes recorded.',
        '',
        'REVIEW LIMITATIONS',
        '------------------',
        '- Research use only. Not a medical device and not a diagnosis.',
        '- Model scores compare predefined positive/negative prompts and are not calibrated disease probabilities.',
        '- Trained for contrast-enhanced abdominal CT; results on other protocols or body regions may be unreliable.',
        '- This draft was auto-generated and requires qualified radiologist review before clinical use.',
        '- Display parameters and anatomy overlays support review only and are not diagnostic-grade.',
    ])
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Clinical validation: compare model scores against a radiologist reference.
# ---------------------------------------------------------------------------

# Term aliases used when extracting findings from a pasted radiologist report.
# Organ name -> alternative terms that should count as mentioning the organ.
_ORGAN_ALIASES = {
    'liver': {'liver', 'hepatic'},
    'kidney': {'kidney', 'renal'},
    'large bowel': {'large bowel', 'colon'},
    'small bowel': {'small bowel', 'small intestine'},
    'adrenal gland': {'adrenal'},
    'gallbladder': {'gallbladder', 'biliary'},
    'portal vein': {'portal vein'},
    'heart': {'heart', 'cardiac'},
    'lung': {'lung', 'pulmonary'},
    'stomach': {'stomach', 'gastric'},
    'esophagus': {'esophagus', 'oesophagus'},
    'bladder': {'bladder'},
    'duodenum': {'duodenum'},
    'pancreas': {'pancreas'},
    'spleen': {'spleen'},
    'aorta': {'aorta'},
}

_FINDING_ALIASES = {
    'cyst': {'cyst', 'cysts'},
    'stone': {'stone', 'stones', 'calculus', 'calculi'},
    'cholecystolithiasis': {'gallstone', 'gallstones', 'cholecystolithiasis'},
    'steatotic liver disease': {'steatosis', 'fatty liver', 'steatotic'},
    'hydronephrosis': {'hydronephrosis', 'hydro'},
    'splenomegaly': {'splenomegaly', 'enlarged spleen'},
    'aortic dissection': {'dissection'},
    'aneurysm': {'aneurysm', 'aneurysms'},
    'fracture': {'fracture', 'fractures'},
    'diverticulum': {'diverticulosis', 'diverticulum', 'diverticula'},
    'obstruction': {'obstruction', 'obstructing'},
    'metastasis': {'metastasis', 'metastases'},
    'lymphoma': {'lymphoma'},
    'pericardial effusion': {'pericardial effusion'},
    'pleural effusion': {'pleural effusion'},
    'pneumothorax': {'pneumothorax'},
    'appendicitis': {'appendicitis'},
    'cirrhosis': {'cirrhosis'},
    'ulcer': {'ulcer', 'ulcers'},
}


def match_findings_to_report(text, scores):
    """Suggest finding keys mentioned in a radiologist report text.

    A finding matches when its finding term (or an alias) appears in the
    text. Ambiguous terms (e.g. "cyst" matches several organs) require the
    organ to also be mentioned; unique terms (e.g. "gallstones") match on
    their own. Results are candidates, not decisions.
    """
    hay = (text or '').lower()
    labels = _finding_labels(scores)
    key_info = []
    for key, label in labels.items():
        organ = label.split('/', 1)[0].strip()
        finding = label.split('/', 1)[1].strip()
        organ_terms = {organ.lower()} | _ORGAN_ALIASES.get(organ.lower(), set())
        finding_terms = {finding.lower()} | _FINDING_ALIASES.get(finding.lower(), set())
        key_info.append((key, organ, organ_terms, finding_terms))
    term_counts = Counter()
    for _, _, _, finding_terms in key_info:
        for term in finding_terms:
            term_counts[term] += 1
    matched = []
    for key, organ, organ_terms, finding_terms in key_info:
        hit_terms = [term for term in finding_terms if term in hay]
        if not hit_terms:
            continue
        organ_hit = any(term in hay for term in organ_terms)
        ambiguous = any(term_counts[term] > 1 for term in hit_terms)
        if organ_hit or not ambiguous:
            matched.append(key)
    return matched


def validation_table(record, threshold=REPORT_SCORE_THRESHOLD):
    """Per-finding agreement rows for adjudicated findings.

    Reference comes from record['validation']['present'/'absent']; a model
    prediction is positive when the score is not missing and >= threshold.
    """
    ref = record.get('validation', {}) or {}
    present = set(ref.get('present', []))
    absent = set(ref.get('absent', []))
    adjudicated = present | absent
    if not adjudicated:
        return pd.DataFrame(columns=['key', 'finding', 'score', 'reference',
                                     'predicted', 'agreement'])
    rows = score_table(record.get('scores', {}))
    row_lookup = rows.set_index('key').to_dict('index')
    labels = _finding_labels(record.get('scores', {}))
    out = []
    for key in sorted(adjudicated):
        row = row_lookup.get(key)
        if row is None:
            continue
        score = row.get('score')
        truth = 1 if key in present else 0
        pred = 1 if score is not None and not pd.isna(score) and score >= threshold else 0
        agreement = ('TP' if pred and truth else
                     'FP' if pred and not truth else
                     'FN' if truth and not pred else 'TN')
        out.append({
            'key': key,
            'finding': labels.get(key, key),
            'score': None if pd.isna(score) else float(score),
            'reference': 'Present' if truth else 'Absent',
            'predicted': 'Positive' if pred else 'Negative',
            'agreement': agreement,
        })
    return pd.DataFrame(out, columns=['key', 'finding', 'score', 'reference',
                                      'predicted', 'agreement'])


def validation_summary(record, threshold=REPORT_SCORE_THRESHOLD):
    """Confusion-matrix counts and derived metrics for adjudicated findings."""
    table = validation_table(record, threshold=threshold)
    if table.empty:
        return {'n': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                'accuracy': None, 'sensitivity': None, 'specificity': None,
                'precision': None, 'f1': None}
    counts = {
        'tp': int((table['agreement'] == 'TP').sum()),
        'fp': int((table['agreement'] == 'FP').sum()),
        'tn': int((table['agreement'] == 'TN').sum()),
        'fn': int((table['agreement'] == 'FN').sum()),
    }
    n = len(table)

    def ratio(num, den):
        return num / den if den else None

    sensitivity = ratio(counts['tp'], counts['tp'] + counts['fn'])
    specificity = ratio(counts['tn'], counts['tn'] + counts['fp'])
    precision = ratio(counts['tp'], counts['tp'] + counts['fp'])
    accuracy = ratio(counts['tp'] + counts['tn'], n)
    f1 = None
    if sensitivity is not None and precision is not None and (sensitivity + precision):
        f1 = 2 * sensitivity * precision / (sensitivity + precision)
    return {'n': n, **counts, 'accuracy': accuracy, 'sensitivity': sensitivity,
            'specificity': specificity, 'precision': precision, 'f1': f1}


def validation_report(record, threshold=REPORT_SCORE_THRESHOLD, total_findings=146):
    """Text summary of RADAR-vs-radiologist agreement for a case."""
    summary = validation_summary(record, threshold=threshold)
    table = validation_table(record, threshold=threshold)

    def fmt(value):
        return f'{value:.3f}' if value is not None else 'n/a (insufficient data)'

    lines = [
        'VALIDATION SUMMARY',
        '------------------',
        f'Case: {record.get("file_name", "unknown_case")}',
        f'Adjudicated findings: {summary["n"]} (of {total_findings} predefined)',
        f'Model prediction threshold: {threshold:.2f}',
        '',
        f'Agreement: TP {summary["tp"]} \u2022 FP {summary["fp"]} \u2022 '
        f'TN {summary["tn"]} \u2022 FN {summary["fn"]}',
        f'Accuracy: {fmt(summary["accuracy"])}',
        f'Sensitivity (recall): {fmt(summary["sensitivity"])}',
        f'Specificity: {fmt(summary["specificity"])}',
        f'Precision (PPV): {fmt(summary["precision"])}',
        f'F1: {fmt(summary["f1"])}',
        '',
        'Per-finding agreement:',
    ]
    if table.empty:
        lines.append('  No findings adjudicated yet.')
    else:
        for _, row in table.iterrows():
            score_text = 'n/a' if row['score'] is None else f'{row["score"]:.3f}'
            lines.append(f'  {row["finding"]}: score {score_text}, '
                         f'reference {row["reference"]}, predicted {row["predicted"]} \u2014 {row["agreement"]}')
    return '\n'.join(lines)

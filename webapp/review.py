"""Data operations for the CT review workspace, independent of Streamlit."""
import shutil
from pathlib import Path
import pandas as pd


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
        if key in ('work_dir', 'result', 'review_notes', 'review_flags', 'review_status') or key.startswith(('slice_', 'selected_')):
            state.pop(key, None)

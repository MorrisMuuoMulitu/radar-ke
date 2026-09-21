"""Data operations for the CT review workspace, independent of Streamlit."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
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


def history_dir():
    return Path(os.environ.get('RADAR_CASE_HISTORY_DIR', Path.home() / '.radar_ke' / 'cases'))


def _safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def _case_id(file_name, saved_at):
    seed = f'{file_name}|{saved_at}'.encode('utf-8')
    return hashlib.sha256(seed).hexdigest()[:16]


def build_case_record(result, status, shortlist, notes, saved_at=None):
    saved_at = saved_at or datetime.now(timezone.utc).isoformat()
    scores = {key: _safe_float(value) for key, value in result.get('scores', {}).items()}
    file_name = result.get('file_name', 'unknown_case')
    return {
        'case_id': _case_id(file_name, saved_at),
        'file_name': file_name,
        'saved_at': saved_at,
        'status': status,
        'shortlist': list(shortlist or []),
        'notes': notes or '',
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


def structured_report(record):
    rows = score_table(record.get('scores', {}))
    labels = dict(zip(rows.key, rows.organ + ' / ' + rows.finding))
    lines = [
        'RADAR Structured Review Draft',
        '',
        f'Case: {record.get("file_name", "unknown_case")}',
        f'Review status: {record.get("status", "Not started")}',
        f'Saved at: {record.get("saved_at", "")}',
        '',
        'Shortlisted findings:',
    ]
    shortlist = record.get('shortlist', [])
    if shortlist:
        scores = record.get('scores', {})
        for key in shortlist:
            score = scores.get(key)
            score_text = 'unavailable' if score is None or pd.isna(score) else f'{float(score):.3f}'
            lines.append(f'- {labels.get(key, key)}: {score_text}')
    else:
        lines.append('- None selected')
    lines.extend([
        '',
        'Reviewer notes:',
        record.get('notes', '').strip() or 'No notes recorded.',
        '',
        'Notice: Research use only. This draft is not a diagnosis and requires qualified radiologist review.',
    ])
    return '\n'.join(lines)

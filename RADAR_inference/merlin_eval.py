"""MERLIN external-evaluation helpers.

Pure Python/NumPy (no torch) so the metric logic is unit-testable and can run
without a GPU or the dataset. Mirrors the protocol of
`calc_metrics_merlin_testset.py`:

* labels are ``{merlin_disease: {patient_id: 0|1|-1}}``; ``-1`` means "not
  evaluated" following the MERLIN protocol and is skipped;
* missing model scores are treated as ``0`` (organ not intact);
* ``surgically_absent_gallbladder`` is derived from segmentation
  (score < 1000 => absent).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

# RADAR finding key -> MERLIN disease label (same mapping as the official script).
RADAR_TO_MERLIN = {
    '主动脉_主动脉瘤': 'abdominal_aortic_aneurysm',
    '主动脉_粥样硬化': 'atherosclerosis',
    '大肠_粘膜下水肿': 'submucosal_edema',
    '大肠_阑尾炎': 'appendicitis',
    '小肠_梗阻': 'bowel_obstruction',
    '心脏_主动脉瓣钙化': 'aortic_valve_calcification',
    '心脏_心影（脏）增大': 'cardiomegaly',
    '肝_肝内胆管扩张': 'biliary_ductal_dilation',
    '肝_肝大': 'hepatomegaly',
    '肝_脂肪肝': 'hepatic_steatosis',
    '肺_胸腔积液': 'pleural_effusion',
    '肺_膨胀不全': 'atelectasis',
    '肾_低密度影': 'renal_hypodensities',
    '肾_囊肿': 'renal_cyst',
    '肾_肾积水': 'hydronephrosis',
    '胆囊_结石': 'gallstones',
    '胰腺_萎缩': 'pancreatic_atrophy',
    '脾_脾大': 'splenomegaly',
    '腰椎_骨折': 'fracture',
    '食管_裂孔疝': 'hiatal_hernia',
    '胆囊_术后胆囊缺失': 'surgically_absent_gallbladder',
}

# Published results (docs/INFERENCE.md, RADAR pretrained on RAD-CT, external MERLIN test set).
PUBLISHED_AUCS = {
    'abdominal_aortic_aneurysm': 0.9903,
    'atherosclerosis': 0.8739,
    'submucosal_edema': 0.8879,
    'appendicitis': 0.7621,
    'bowel_obstruction': 0.9704,
    'aortic_valve_calcification': 0.8436,
    'cardiomegaly': 0.8724,
    'biliary_ductal_dilation': 0.8711,
    'hepatomegaly': 0.8988,
    'hepatic_steatosis': 0.8917,
    'pleural_effusion': 0.9574,
    'atelectasis': 0.7091,
    'renal_hypodensities': 0.9122,
    'renal_cyst': 0.9426,
    'hydronephrosis': 0.88,
    'gallstones': 0.9193,
    'pancreatic_atrophy': 0.9356,
    'splenomegaly': 0.9682,
    'fracture': 0.6834,
    'hiatal_hernia': 0.8601,
    'surgically_absent_gallbladder': 0.9234,
}

PUBLISHED_AVG_AUC = 0.8835

# Seconds per case measured in this deployment (RTX 4060, compact windows).
SECONDS_PER_CASE = 26


def auc_score(y_true, y_score):
    """Tie-aware ROC AUC (Mann-Whitney U). Returns None if a class is absent."""
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs)
    if n == 0:
        return None
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def select_subset(patient_ids, n=None, seed=0):
    """Deterministic random subset of case ids (sorted before sampling)."""
    ids = sorted({str(pid) for pid in patient_ids})
    if n is None or n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, int(n)))


def patient_id(file_name):
    """MERLIN file name -> patient id (strips the .nii.gz suffix)."""
    name = str(file_name)
    for suffix in ('.nii.gz', '.nii'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_labels(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def radar_key(column):
    """Normalise a results column to the bare RADAR finding key.

    Inference writes columns as ``主动脉_主动脉瘤_(abdominal_aortic_aneurysm)``;
    the metric mapping is keyed by the bare Chinese name.
    """
    name = str(column)
    marker = name.find('_(')
    return name[:marker] if marker != -1 else name


def compute_aucs(results, labels, mapping=None):
    """Per-disease AUC from a results table (``file_name`` + finding columns)."""
    mapping = mapping or RADAR_TO_MERLIN
    computed = {}
    for column in list(results.columns):
        if column == 'file_name':
            continue
        key = radar_key(column)
        disease = mapping.get(key)
        if disease is None:
            continue
        disease_labels = labels.get(disease, {})
        y_true, y_score = [], []
        for file_name, probability in zip(results['file_name'], results[column]):
            label = disease_labels.get(patient_id(file_name))
            if label is None:
                continue
            try:
                label = float(label)
            except (TypeError, ValueError):
                continue
            if label == -1:                      # MERLIN protocol: not evaluated
                continue
            score = 0.0 if probability is None or (
                isinstance(probability, float) and math.isnan(probability)) else float(probability)
            y_true.append(label)
            y_score.append(score)
        if key == '胆囊_术后胆囊缺失':            # segmentation-derived presence
            y_score = [0.0 if value < 1000 else 1.0 for value in y_score]
        computed[disease] = {
            'auc': auc_score(y_true, y_score),
            'n': len(y_true),
            'positives': int(sum(y_true)),
        }
    return computed


def compare_to_published(computed):
    """Rows comparing computed AUCs with the published values."""
    rows = []
    for disease, published in PUBLISHED_AUCS.items():
        entry = computed.get(disease) or {}
        value = entry.get('auc')
        rows.append({
            'disease': disease,
            'published': published,
            'computed': value,
            'delta': None if value is None else value - published,
            'n': entry.get('n', 0),
            'positives': entry.get('positives', 0),
        })
    return rows


def mean_auc(computed):
    values = [entry['auc'] for entry in computed.values() if entry.get('auc') is not None]
    return sum(values) / len(values) if values else None


def format_report(rows, computed, n_cases, results_csv=None, title='MERLIN subset evaluation'):
    """Markdown report: per-disease comparison plus the average AUC."""
    lines = [f'# {title}', '']
    if results_csv:
        lines.append(f'Results: `{results_csv}`  ')
    lines.append(f'Cases scored: **{n_cases}**  ')
    average = mean_auc(computed)
    if average is not None:
        lines.append(f'Average AUC: **{average:.4f}** (published {PUBLISHED_AVG_AUC:.4f})')
    lines += ['', '| finding | published | computed | delta | n | positives |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        computed_text = '—' if row['computed'] is None else f"{row['computed']:.4f}"
        delta_text = '—' if row['delta'] is None else f"{row['delta']:+.4f}"
        lines.append(f"| {row['disease']} | {row['published']:.4f} | {computed_text} | "
                     f"{delta_text} | {row['n']} | {row['positives']} |")
    lines.append('')
    lines.append('Note: reproduce on this deployment; not a new clinical claim.')
    return '\n'.join(lines)


def list_case_files(data_dir, suffixes=('.nii.gz', '.nii')):
    """Case volumes present in a directory."""
    root = Path(data_dir)
    return sorted(p for p in root.iterdir() if p.is_file() and p.name.endswith(suffixes))


def estimate(data_dir, n=None, seconds_per_case=SECONDS_PER_CASE):
    """Sizes and runtime estimate for a subset selection."""
    files = list_case_files(data_dir)
    total_bytes = sum(p.stat().st_size for p in files)
    per_case = total_bytes / len(files) if files else 0
    selected = min(n, len(files)) if n else len(files)
    return {
        'available': len(files),
        'selected': selected,
        'bytes_selected': int(per_case * selected),
        'est_seconds': int(seconds_per_case * selected),
    }


# ---------------------------------------------------------------------------
# Portal ingestion: files downloaded from the Stanford AIMI MERLIN dataset.
# ---------------------------------------------------------------------------

def build_report_json(xlsx_path, out_path, columns=None):
    """reports_final.xlsx -> merlin_report.json (same shape as ckpt/transform_report_to_json.py)."""
    import pandas as pd
    columns = columns or {'id': 'study id', 'report': 'Findings', 'split': 'Split', 'fewshot': 'Few Shot'}
    frame = pd.read_excel(xlsx_path)
    info = {}
    for pid, report, split, fewshot in zip(frame[columns['id']], frame[columns['report']],
                                           frame[columns['split']], frame[columns['fewshot']]):
        report = '' if report is None else str(report)
        marker = report.find('IMPRESSION:')
        findings, impression = (report[:marker], report[marker:]) if marker != -1 else (report, '')
        info.setdefault(str(pid), {'report': report, 'findings': findings,
                                   'impression': impression, 'split': split, 'fewshot': fewshot})
    Path(out_path).write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding='utf-8')
    return info


def build_labels_json(csv_path, out_path):
    """zero_shot_findings_disease_cls.csv -> merlin_labels.json (disease -> {pid: 0|1|-1})."""
    import pandas as pd
    frame = pd.read_csv(csv_path)
    names = list(frame.columns)
    ids = frame[names[0]]
    labels = {disease: {str(ids[i]): int(frame[disease][i]) for i in range(len(ids))}
              for disease in names[1:]}
    Path(out_path).write_text(json.dumps(labels, ensure_ascii=False, indent=4), encoding='utf-8')
    return labels


def test_split_ids(report_info):
    """Patient ids marked ``split: test`` in merlin_report.json."""
    return sorted(pid for pid, info in report_info.items()
                  if str(info.get('split', '')).strip().lower() == 'test')


def find_volume_files(root, suffixes=('.nii.gz',)):
    """Recursively locate CT volumes (the portal nests them in subfolders)."""
    base = Path(root)
    return sorted(p for p in base.rglob('*') if p.is_file() and p.name.endswith(suffixes))


def match_volumes(patient_ids, volume_files):
    """Map patient id -> volume path for ids that are present on disk."""
    by_id = {patient_id(path.name): path for path in volume_files}
    return {pid: by_id[pid] for pid in patient_ids if pid in by_id}

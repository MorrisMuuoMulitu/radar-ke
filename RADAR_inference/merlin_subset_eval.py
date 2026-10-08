"""Run RADAR on a MERLIN subset and compare AUCs with the published results.

Designed for the "register, download a sample, evaluate, delete" workflow:

  # 1) plan (no GPU, no data): pick cases and see the size/time estimate
  python RADAR_inference/merlin_subset_eval.py --data-dir /data/merlin_test \\
      --n 200 --seed 0 --out results/merlin_subset --dry-run

  # 2) score the selected cases (GPU), append to a master results CSV
  python RADAR_inference/merlin_subset_eval.py --data-dir /data/merlin_test \\
      --n 200 --seed 0 --out results/merlin_subset \\
      --labels ckpt/merlin_labels.json --results results/merlin_subset/results.csv

  # 3) recompute metrics only (no GPU) from an existing results CSV
  python RADAR_inference/merlin_subset_eval.py --metrics-only <results.csv> \\
      --labels ckpt/merlin_labels.json --out results/merlin_subset

Re-runs are resumable: cases already present in --results are skipped.  This
mirrors the official `inference_merlin_testset.py` pipeline (21 MERLIN items,
`infer_text_embedding_merlin.pt`) so the numbers are comparable with
`docs/INFERENCE.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import merlin_eval  # noqa: E402

os.environ.setdefault('MODEL_ROOT', str(REPO_ROOT / 'ckpt'))
os.environ.setdefault('CONFIGS_ROOT', str(REPO_ROOT / 'ckpt'))
# Fragmentation-friendly allocator; on <=10 GB GPUs also pass ROI_SIZE=64,192,288
# (full-size 96x256x384 windows need more VRAM than an 8 GB card has).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def resolve_selection(args):
    """Case ids selected for scoring (explicit list, random subset, or all)."""
    files = merlin_eval.list_case_files(args.data_dir)
    available = [f.name for f in files]
    if args.list_file:
        wanted = [line.strip() for line in Path(args.list_file).read_text().splitlines() if line.strip()]
        selected = [name for name in available if name in set(wanted)]
    elif args.n:
        selected = merlin_eval.select_subset(available, args.n, args.seed)
    else:
        selected = sorted(available)
    return files, available, selected


def already_scored(results_csv):
    if results_csv and Path(results_csv).exists():
        return set(pd.read_csv(results_csv)['file_name'].astype(str))
    return set()


def write_selection(out_dir, selected):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'selection.txt'
    path.write_text('\n'.join(selected) + '\n', encoding='utf-8')
    return path


def stage_cases(files, selected, staging):
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    wanted = set(selected)
    for path in files:
        if path.name in wanted:
            (staging / path.name).symlink_to(path.resolve())


def merge_results(new_csv, master_csv):
    """Append newly scored rows to the master CSV (deduplicated by file_name)."""
    new = pd.read_csv(new_csv)
    if master_csv.exists():
        master = pd.read_csv(master_csv)
        combined = pd.concat([master, new], ignore_index=True)
        combined = combined.drop_duplicates(subset='file_name', keep='last')
    else:
        combined = new
    combined.to_csv(master_csv, index=False, encoding='utf-8-sig')
    return combined


def run_metrics(results_csv, labels_path, out_dir, n_cases=None):
    results = pd.read_csv(results_csv)
    labels = merlin_eval.load_labels(labels_path) if labels_path else {}
    computed = merlin_eval.compute_aucs(results, labels)
    rows = merlin_eval.compare_to_published(computed)
    report = merlin_eval.format_report(rows, computed, n_cases or len(results), str(results_csv))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'merlin_comparison.md').write_text(report, encoding='utf-8')
    (out_dir / 'merlin_comparison.json').write_text(
        json.dumps({'n_cases': n_cases or len(results),
                    'mean_auc': merlin_eval.mean_auc(computed),
                    'published_mean_auc': merlin_eval.PUBLISHED_AVG_AUC,
                    'rows': rows}, indent=2), encoding='utf-8')
    return report


def preflight(labels_path=None):
    """Fail early with actionable messages about the required MERLIN inputs."""
    problems = []
    report = Path(os.environ.get('CONFIGS_ROOT', REPO_ROOT / 'ckpt')) / 'merlin_report.json'
    if not report.exists():
        problems.append(f'• missing {report}\n'
                        '  → generate it from the portal reports_final.xlsx: ckpt/transform_report_to_json.py')
    if labels_path and not Path(labels_path).exists():
        problems.append(f'• missing labels file {labels_path}\n'
                        '  → generate it from the portal zero_shot_findings_disease_cls.csv: '
                        'ckpt/transform_label_to_json.py')
    if problems:
        print('Cannot run the MERLIN evaluation — required inputs are missing:\n' + '\n'.join(problems))
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', help='Folder of MERLIN case volumes (.nii/.nii.gz).')
    parser.add_argument('--labels', help='merlin_labels.json (from ckpt/transform_label_to_json.py).')
    parser.add_argument('--out', default=str(REPO_ROOT / 'results' / 'merlin_subset'), help='Output folder.')
    parser.add_argument('--n', type=int, default=None, help='Number of cases to sample.')
    parser.add_argument('--seed', type=int, default=0, help='Sampling seed (deterministic selection).')
    parser.add_argument('--list-file', help='Explicit case list (one file name per line).')
    parser.add_argument('--results', help='Master results CSV to append to / resume from.')
    parser.add_argument('--save-tag', default='MerlinSubset', help='Save tag for the inference CSV.')
    parser.add_argument('--dry-run', action='store_true', help='Plan only: write selection + estimates, no inference.')
    parser.add_argument('--metrics-only', help='Skip inference; compute metrics from this results CSV.')
    parser.add_argument('--delete-source', action='store_true',
                        help='Delete each source volume after it is scored (bounded-disk chunking).')
    args = parser.parse_args()

    out_dir = Path(args.out)
    master_csv = Path(args.results) if args.results else out_dir / 'results.csv'

    if args.metrics_only:
        report = run_metrics(args.metrics_only, args.labels, out_dir)
        print(report)
        return 0

    if not args.data_dir:
        parser.error('--data-dir is required unless --metrics-only is used')

    files, available, selected = resolve_selection(args)
    if not selected:
        print(f'No matching case volumes in {args.data_dir}')
        return 1
    selection_path = write_selection(out_dir, selected)
    per_case = sum(f.stat().st_size for f in files) / len(files) if files else 0
    print(f'Available: {len(available)} | selected: {len(selected)} | '
          f'~{per_case * len(selected) / 1e9:.1f} GB | ~{26 * len(selected) / 3600:.1f} GPU-hours')
    print(f'Selection written to {selection_path}')

    if args.dry_run:
        return 0

    if not preflight(args.labels):
        return 1

    done = already_scored(master_csv)
    todo = [name for name in selected if name not in done]
    print(f'Already scored: {len(done & set(selected))} | to score: {len(todo)}')
    if not todo:
        if args.labels:
            print(run_metrics(master_csv, args.labels, out_dir, n_cases=len(selected)))
        return 0

    staging = out_dir / 'staging'
    stage_cases(files, todo, staging)

    from inference_merlin_testset import evaluate, initialize   # torch import, GPU path
    pad_func, model = initialize()
    evaluate(pad_func, model, str(staging), str(out_dir), args.save_tag)

    produced = out_dir / f'RADAR_infer_results_{args.save_tag}.csv'
    combined = merge_results(produced, master_csv)
    print(f'Master results: {master_csv} ({len(combined)} cases)')

    if args.delete_source:
        for path in files:
            if path.name in set(todo):
                path.unlink()
        print(f'Deleted {len(todo)} source volumes (--delete-source)')

    if args.labels:
        print(run_metrics(master_csv, args.labels, out_dir, n_cases=len(selected)))
    else:
        print('No --labels given: results CSV written, metrics skipped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

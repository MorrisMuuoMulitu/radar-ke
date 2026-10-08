"""One-command MERLIN evaluation for this deployment.

From the folder you downloaded from the Stanford AIMI portal, this:

1. locates ``reports_final.xlsx``, ``zero_shot_findings_disease_cls.csv`` and the
   CT volumes (the portal nests them in subfolders — no need to flatten);
2. generates ``ckpt/merlin_report.json`` / ``ckpt/merlin_labels.json`` if missing;
3. matches the official **test split** against the volumes you have;
4. scores a deterministic sample (or all of them) in memory-bounded chunks,
   resuming from an existing results CSV;
5. writes a computed-vs-published AUC comparison.

Examples
--------
    # readiness check (no GPU work)
    python RADAR_inference/merlin_run.py --portal-dir ~/Downloads/merlinabdominalctdataset --check

    # score 200 test studies and compare with the published AUCs
    MODEL_ROOT=$PWD/ckpt CONFIGS_ROOT=$PWD/ckpt ROI_SIZE=64,192,288 \
    python RADAR_inference/merlin_run.py --portal-dir ~/Downloads/merlinabdominalctdataset --n 200

    # every test study you have, deleting each source after it is scored
    python RADAR_inference/merlin_run.py --portal-dir ... --all --delete-source
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import merlin_eval  # noqa: E402
import merlin_subset_eval as subset  # noqa: E402  (sets MODEL_ROOT/CONFIGS_ROOT defaults)

CKPT_DIR = Path(os.environ.get('CONFIGS_ROOT', REPO_ROOT / 'ckpt'))


def find_one(root, name):
    matches = sorted(Path(root).rglob(name))
    return matches[0] if matches else None


def locate_portal_inputs(portal_dir):
    return {
        'reports': find_one(portal_dir, 'reports_final.xlsx'),
        'labels': find_one(portal_dir, 'zero_shot_findings_disease_cls.csv'),
        'volumes': merlin_eval.find_volume_files(portal_dir),
    }


def ensure_jsons(found, rebuild=False):
    report_path = CKPT_DIR / 'merlin_report.json'
    labels_path = CKPT_DIR / 'merlin_labels.json'
    report_info = None
    if found['reports'] and (rebuild or not report_path.exists()):
        report_info = merlin_eval.build_report_json(found['reports'], report_path)
        print(f'  generated {report_path} ({len(report_info)} studies)')
    elif report_path.exists():
        report_info = json.loads(report_path.read_text(encoding='utf-8'))
    if found['labels'] and (rebuild or not labels_path.exists()):
        labels = merlin_eval.build_labels_json(found['labels'], labels_path)
        print(f'  generated {labels_path} ({len(labels)} diseases)')
    return report_info, report_path, labels_path


def readiness(args):
    portal = Path(args.portal_dir).expanduser()
    if not portal.exists():
        print(f'Portal folder not found: {portal}')
        return None
    found = locate_portal_inputs(portal)
    print(f'Portal folder: {portal}')
    print(f"  reports_final.xlsx                 : {found['reports'] or 'MISSING'}")
    print(f"  zero_shot_findings_disease_cls.csv : {found['labels'] or 'MISSING'}")
    print(f"  volumes found (*.nii.gz)           : {len(found['volumes'])}")

    report_info, report_path, labels_path = ensure_jsons(found, args.rebuild_json)
    if report_info is None:
        print(f'\nNo test-split information: {report_path} is missing and reports_final.xlsx was not found.')
        return None
    test_ids = merlin_eval.test_split_ids(report_info)
    matched = merlin_eval.match_volumes(test_ids, found['volumes'])
    print(f'  test-split studies in report       : {len(test_ids)}')
    print(f'  test-split volumes present on disk : {len(matched)}')
    print(f"  labels json                        : {labels_path if labels_path.exists() else 'MISSING'}")

    if not matched:
        print('\nNo test-split volumes found — check --portal-dir or the download.')
        return None
    # Selection/staging/resume all work on file names (what the inference CSV records).
    paths_by_name = {path.name: path for path in matched.values()}
    names = sorted(paths_by_name)
    selected = names if args.all else merlin_eval.select_subset(names, args.n, args.seed)
    per_case = sum(paths_by_name[name].stat().st_size for name in selected) / max(1, len(selected))
    free = shutil.disk_usage(REPO_ROOT).free
    gpu = 'unknown'
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
    except Exception:
        pass
    print('\nPlan')
    print(f'  selected                           : {len(selected)} studies (seed {args.seed})')
    print(f'  source size of selection           : {per_case * len(selected) / 1e9:.1f} GB')
    print(f'  est. GPU time                      : {26 * len(selected) / 3600:.1f} h (8 GB card, compact windows)')
    print(f'  free disk                          : {free / 1e9:.1f} GB')
    print(f'  compute device                     : {gpu}')
    if args.chunk:
        print(f'  chunk size                         : {args.chunk} studies per pass')
    return {'found': found, 'report_info': report_info, 'matched': matched,
            'paths_by_name': paths_by_name, 'selected': selected, 'labels_path': labels_path}


def score(args, plan):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = Path(args.results) if args.results else out_dir / 'results.csv'
    selected = plan['selected']
    subset.write_selection(out_dir, selected)

    done = subset.already_scored(master_csv)
    todo = [name for name in selected if name not in done]
    print(f"\nAlready scored: {len(set(selected) & done)} | to score: {len(todo)}")
    if not todo:
        if plan['labels_path'].exists():
            print(subset.run_metrics(master_csv, plan['labels_path'], out_dir, n_cases=len(selected)))
        return 0

    chunk_size = args.chunk or len(todo)
    staging = out_dir / 'staging'
    from inference_merlin_testset import evaluate, initialize
    pad_func, model = initialize()
    started = time.monotonic()
    scored = 0
    for index in range(0, len(todo), chunk_size):
        chunk = todo[index:index + chunk_size]
        print(f'\n--- chunk {index // chunk_size + 1}: {len(chunk)} studies ---')
        subset.stage_cases(plan['found']['volumes'], chunk, staging)
        tag = f'{args.save_tag}_chunk{index // chunk_size + 1}'
        evaluate(pad_func, model, str(staging), str(out_dir), tag)
        subset.merge_results(out_dir / f'RADAR_infer_results_{tag}.csv', master_csv)
        (out_dir / f'RADAR_infer_results_{tag}.csv').unlink(missing_ok=True)
        scored += len(chunk)
        if args.delete_source:
            for name in chunk:
                Path(plan['paths_by_name'][name]).unlink(missing_ok=True)
            print(f'    deleted {len(chunk)} source volumes (--delete-source)')
    elapsed = time.monotonic() - started
    print(f'\nScored {scored} studies in {elapsed / 60:.1f} min '
          f'({elapsed / max(1, scored):.1f} s/study) -> {master_csv}')

    if plan['labels_path'].exists():
        report = subset.run_metrics(master_csv, plan['labels_path'], out_dir, n_cases=len(selected))
        print('\n' + report)
    else:
        print('No merlin_labels.json: results CSV written, metrics skipped.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--portal-dir', required=True, help='Folder downloaded from the Stanford AIMI portal.')
    parser.add_argument('--out', default=str(REPO_ROOT / 'results' / 'merlin_subset'), help='Output folder.')
    parser.add_argument('--n', type=int, default=200, help='Studies to sample (default 200).')
    parser.add_argument('--all', action='store_true', help='Use every test-split study present.')
    parser.add_argument('--seed', type=int, default=0, help='Sampling seed.')
    parser.add_argument('--chunk', type=int, default=50, help='Studies per inference pass (memory/disk bound).')
    parser.add_argument('--results', help='Master results CSV (default <out>/results.csv).')
    parser.add_argument('--save-tag', default='MerlinSubset')
    parser.add_argument('--check', action='store_true', help='Readiness report only (no inference).')
    parser.add_argument('--dry-run', action='store_true', help='Write the selection and estimates, no inference.')
    parser.add_argument('--rebuild-json', action='store_true', help='Regenerate the ckpt JSONs from the portal files.')
    parser.add_argument('--delete-source', action='store_true',
                        help='Delete each source volume after scoring it (destructive; bounded disk).')
    args = parser.parse_args()

    print('=== MERLIN evaluation readiness ===')
    plan = readiness(args)
    if plan is None:
        return 1
    if args.check:
        return 0
    if args.dry_run:
        out_dir = Path(args.out)
        subset.write_selection(out_dir, plan['selected'])
        print(f"\nSelection written to {out_dir / 'selection.txt'} (dry run — nothing scored).")
        return 0
    if not subset.preflight(plan['labels_path'] if plan['labels_path'].exists() else None):
        return 1
    return score(args, plan)


if __name__ == '__main__':
    raise SystemExit(main())

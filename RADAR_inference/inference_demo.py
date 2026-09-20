"""Thin CLI wrapper around inference_service (original inference_demo behavior).

Usage:
    python inference_demo.py --img_dir ../data/demo_cases --save_dir ../results --save_tag demo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_service import run_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../data/demo_cases', help='The path to inference image folder.')
    parser.add_argument('--save_dir', type=str, default='../results', help='The path to save folder.')
    parser.add_argument('--save_tag', type=str, default='demo', help='Save tag.')
    parser.add_argument('--collect-cases', action='store_true',
                        help='Also save per-case .npz (image/mask/scores) for the web viewer.')
    parser.add_argument('--case_dir', type=str, default=None,
                        help='Directory for per-case .npz files (defaults to <save_dir>/cases).')
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = args.case_dir or os.path.join(args.save_dir, 'cases')
    run_folder(args.img_dir, args.save_dir, args.save_tag,
               collect_cases=args.collect_cases,
               case_dir=case_dir if args.collect_cases else None)


if __name__ == '__main__':
    main()

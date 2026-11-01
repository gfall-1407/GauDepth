"""Run only the frozen scan110 ray17/root2 and ray24/root6 CPU VJP audit."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from raw_param_bridge import fixed_root_vjp, load_fixed_capture, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--roots', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output_already_exists')
    root = Path(__file__).resolve().parents[2]
    math_dir = root / '实验' / '20261101_局部形状参考'
    sys.path.insert(0, str(math_dir))
    from local_shape_grad_reference import selected_root

    arrays, rows, metadata = load_fixed_capture(args.capture, args.roots, args.metadata)
    result = fixed_root_vjp(arrays, rows, selected_root)
    result['source'] = dict(capture_sha256=sha256(args.capture),
                            metadata_sha256=sha256(args.metadata),
                            roots_sha256=sha256(args.roots),
                            ply_sha256=metadata['ply_sha256'],
                            math_source_sha256=sha256(math_dir / 'local_shape_grad_reference.py'),
                            local_reference_sha256=sha256(math_dir / 'local_shape_reference.py'),
                            bridge_sha256=sha256(Path(__file__).with_name('raw_param_bridge.py')),
                            runner_sha256=sha256(__file__),
                            git_commit=subprocess.check_output(
                                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), roots=result['roots'],
                          checks=len(result['checks']), shared_ids=len(result['shared_ids']))))


if __name__ == '__main__':
    main()

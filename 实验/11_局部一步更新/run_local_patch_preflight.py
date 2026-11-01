"""Read-only CPU preflight for two fixed 3x3 local-root patches."""
import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / '实验' / '10_原参梯度接线'))
sys.path.insert(0, str(root / '实验' / '20261101_局部形状参考'))

from local_shape_grad_reference import selected_root
from raw_param_bridge import ROOTS, load_fixed_capture, shared_raw_leaves, sha256
from local_patch_root import patch_root_normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--roots', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output_already_exists')
    arrays, rows, _ = load_fixed_capture(args.capture, args.roots, args.metadata)
    raw = shared_raw_leaves(arrays)
    result = dict(capture_sha256=sha256(args.capture), roots_sha256=sha256(args.roots),
                  candidate='same_support_four_substeps_fixed_bracket', rays={})
    for center, _ in ROOTS:
        try:
            z, normal, details = patch_root_normal(arrays, rows[center], raw, selected_root)
            loss = z + normal @ normal.new_tensor([.2, -.3, .4])
            import torch
            grads = torch.autograd.grad(loss, [raw[name] for name in ('xyz', 'log_scale', 'raw_quat', 'logit')])
            if not all(bool(torch.isfinite(value).all()) for value in grads):
                raise ValueError('nonfinite_local_patch_gradient')
            result['rays'][center] = dict(status='valid_local_patch', z=float(z.detach()),
                                          normal=normal.detach().tolist(),
                                          grad_max=[float(value.abs().max()) for value in grads],
                                          **details)
        except (ValueError, RuntimeError) as exc:
            result['rays'][center] = dict(status='invalid_local_patch', reason=str(exc),
                                          center_z=rows[center]['z'],
                                          bracket=rows[center]['bracket'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), centers=list(result['rays']))))
    if any(row['status'] != 'valid_local_patch' for row in result['rays'].values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

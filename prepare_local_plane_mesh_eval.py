"""Stage four independent Expected/Median mesh evaluation directories after 1000 steps.

This only prepares data and prints commands. It does not invoke GPU rendering.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil

from local_plane_training_cpu import sha256


def stage(run_root,eval_root):
    run_root,eval_root=Path(run_root),Path(eval_root)
    if eval_root.exists() or not run_root.is_dir():raise ValueError("invalid_or_existing_eval_root")
    if not (run_root/"progress.json").is_file():raise ValueError("incomplete_two_arm_training")
    progress=json.loads((run_root/"progress.json").read_text(encoding="utf-8"))
    if set(progress)!={"A","B"} or any(row[-1]["completed_updates"]!=1000 for row in progress.values()):
        raise ValueError("not_two_complete_1000_update_arms")
    plans=[]
    eval_root.mkdir(parents=True)
    for arm in ("A","B"):
        source=run_root/arm/"point_cloud"/"iteration_8000"/"point_cloud.ply"
        cfg=run_root/arm/"cfg_args"
        if not source.is_file() or not cfg.is_file():raise ValueError("missing_final_model_or_cfg")
        from argparse import Namespace
        tree=ast.parse(cfg.read_text(encoding="utf-8"),mode="eval").body
        if not isinstance(tree,ast.Call) or not isinstance(tree.func,ast.Name) or tree.func.id!="Namespace" or tree.args:
            raise ValueError("invalid_cfg_namespace")
        args=Namespace(**{kw.arg:ast.literal_eval(kw.value) for kw in tree.keywords})
        for name,ratio in (("E",0.0),("M",1.0)):
            dest=eval_root/(arm+"-"+name)
            model=dest/"point_cloud"/"iteration_8000"/"point_cloud.ply"
            model.parent.mkdir(parents=True)
            shutil.copyfile(source,model)
            if sha256(model)!=sha256(source):raise ValueError("mesh_input_copy_mismatch")
            options=Namespace(**vars(args));options.model_path=str(dest);options.depth_ratio=ratio
            (dest/"cfg_args").write_text(str(options),encoding="utf-8")
            plans.append(dict(label=arm+"-"+name,source_model_sha256=sha256(source),
                model_dir=str(dest),depth_ratio=ratio,channel="expected_depth" if name=="E" else "median_depth"))
    (eval_root/"mesh_plan.json").write_text(json.dumps(plans,indent=2)+"\n",encoding="utf-8")
    return plans


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-root",required=True)
    parser.add_argument("--eval-root",required=True)
    parser.add_argument("--dtu-root",required=True)
    args=parser.parse_args()
    for item in stage(args.run_root,args.eval_root):
        print(f"python mesh_extract.py -m '{item['model_dir']}' --iteration 8000")
        print(f"python evaluate_dtu_mesh.py -m '{item['model_dir']}' --DTU '{args.dtu_root}'")

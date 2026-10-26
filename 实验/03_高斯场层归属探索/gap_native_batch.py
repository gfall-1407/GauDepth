"""One finite G0 fixed-scan bridge/cache/native-score batch; run under night_guard."""
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from photo_surface_probe import HERE, save, sha256, source_info
from night_depth_batch import FIXED


def main(args):
    if socket.gethostname() != "autodl-container-5d5147b55f-67549ee6":
        raise ValueError("Only the explicitly authorized G0")
    scan=args.scan
    root=args.output.resolve()
    if not args.output.is_absolute() or root.exists():raise ValueError("New absolute result root required")
    root.mkdir(parents=True)
    state=dict(status="partial",scan=scan,source=source_info(),started=time.time(),steps=[])
    def update(status,**kw):state.update(status=status,**kw);save(root/"batch.json",state)
    signal.signal(signal.SIGTERM,lambda *_:(root/"stop_batch").touch())
    def check():
        if (root/"stop_batch").exists():raise InterruptedError("Guard requested saved boundary")
    def task(name,extra):
        check();destination=root/name
        command=[sys.executable,str(HERE/"dense_huber_mesh.py"),"--scan",str(scan),
            "--instance-id","5d5147b55f-67549ee6","--model-root",str(args.model_root),
            "--source-root",str(args.source_root),"--gt-root",str(args.gt_root),"--output",str(destination),*extra]
        update("running",active=name,command=command)
        with (root/(name+".log")).open("xb") as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        meta=json.loads((destination/"metadata.json").read_text(encoding="utf-8")) if (destination/"metadata.json").exists() else {}
        state["steps"].append(dict(name=name,returncode=r.returncode,status=meta.get("status"),metadata=str(destination/"metadata.json")))
        update("step_saved")
        if r.returncode or meta.get("status") not in ("pilot_complete","cache_complete","complete"):
            raise RuntimeError("Failed or partial stage "+name)
        return meta
    def bridge(fresh,old,all_views):
        previous=json.loads((old/"metadata.json").read_text(encoding="utf-8"))
        if previous["status"] not in ("complete","cache_complete"):raise ValueError("Original cache incomplete")
        comparisons=[]
        for view in fresh["views"]:
            prior=next(v for v in previous["views"] if v["name"]==view["name"])
            op=old/prior["file"];new_path=root/(f"cache{scan}" if all_views else f"pilot{scan}")/view["file"]
            if sha256(op)!=prior["sha256"] or sha256(new_path)!=view["sha256"]:raise ValueError("Cache bridge SHA mismatch")
            with np.load(op) as a,np.load(new_path) as b:
                for k in ("rgb","expected","median","alpha","gt_mask"):
                    if not np.array_equal(a[k],b[k],equal_nan=True):raise ValueError("Native bridge differs: "+view["name"]+"/"+k)
            comparisons.append(dict(view=view["name"],original_sha256=prior["sha256"],new_sha256=view["sha256"]))
        if all_views and len(comparisons)!=FIXED[scan][2]:raise ValueError("Need every original camera native bridge")
        result=dict(status="PASS_exact_native_arrays",all_views=all_views,comparisons=comparisons,
            baseline_metadata_sha256=sha256(old/"metadata.json"),baseline_results=previous.get("results",{}))
        save(root/("bridge_all_views.json" if all_views else "bridge_first.json"),result)
    try:
        pilot=task(f"pilot{scan}",["--dense-method","gap_huber","--pilot-only"])
        bridge(pilot,args.original_cache,False)
        check()
        cache=task(f"cache{scan}",["--dense-method","gap_huber","--cache-only"])
        bridge(cache,args.original_cache,True)
        check()
        scores=task(f"native{scan}",["--cache-root",str(root/f"cache{scan}"),"--methods","native_expected","gap_huber"])
        update("complete_waiting_transfer",results=scores.get("results"),finished_compute=time.time())
        # Parent may retrieve results while the independent guard remains armed.
        while not (root/"transfer_verified").exists():check();time.sleep(2)
        update("complete",finished=time.time())
    except BaseException as e:
        update("failed_or_stopped",error=repr(e),finished=time.time());raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("output","model-root","original-cache","source-root","gt-root"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--scan",type=int,choices=tuple(FIXED),default=40)
    main(p.parse_args())

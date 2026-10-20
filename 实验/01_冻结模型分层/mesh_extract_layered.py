"""Use the host extract_mesh body unchanged except depth and output paths."""
import sys
sys.dont_write_bytecode=True
import os
import json
import time
import inspect
from pathlib import Path
from argparse import ArgumentParser
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mesh_extract as host
from arguments import ModelParams,PipelineParams,get_combined_args
from utils.general_utils import safe_state
from layered_depth import depth_maps,protect_scene_output

def run(dataset,pipe,iteration,methods,num_cluster,check_only):
    scan=Path(dataset.source_path).name
    out=HERE/'结果'/scan
    out.mkdir(parents=True,exist_ok=True)
    suffix='_adaptive' if methods==['gap_adaptive'] else ''
    protect_scene_output(out)
    gaussians=host.GaussianModel(dataset.sh_degree)
    scene=host.Scene(dataset,gaussians,load_iteration=iteration,shuffle=False)
    cameras=scene.getTrainCameras()
    manifest=dict(scan=scan,iteration=scene.loaded_iter,resolution=dataset.resolution,
                  depth_ratio=dataset.depth_ratio,num_cluster=num_cluster,
                  cameras=[dict(name=c.image_name,width=c.image_width,height=c.image_height)
                           for c in cameras],methods=methods)
    (out/('运行参数'+suffix+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    background=torch.tensor([1,1,1],dtype=torch.float32,device='cuda')
    frames=[]; statistics=[]; started=time.time()
    for i,cam in enumerate(cameras):
        t=time.time()
        rgb,maps,st=depth_maps(cam,gaussians,pipe,background,dataset.kernel_size,
                               methods,validate=(i==0))
        st.update(view=cam.image_name,seconds=time.time()-t)
        statistics.append(st)
        if check_only:
            (out/'回放核对.json').write_text(json.dumps(st,ensure_ascii=False,indent=2))
            print('REPLAY_PASS',st,flush=True);return
        # Numeric depths stay on the server; no per-contributor records retained.
        frame_dir=out/'深度';frame_dir.mkdir(exist_ok=True)
        depth_file=frame_dir/(cam.image_name+'.npz')
        saved={}
        if depth_file.exists():
            with np.load(depth_file) as previous:
                saved={m:previous[m] for m in previous.files}
        saved.update({m:d.numpy() for m,d in maps.items()})
        np.savez_compressed(depth_file,**saved)
        frames.append((rgb,maps))
        print('VIEW',scan,i+1,len(cameras),st,flush=True)
    (out/('分组统计'+suffix+'.json')).write_text(json.dumps(statistics,ensure_ascii=False,indent=2))
    timings={'depth_seconds':time.time()-started}
    # Reuse the literal host function body, including RGB/mask/TSDF/postprocess.
    body=inspect.getsource(host.extract_mesh)
    body=body.replace('gaussians = GaussianModel(dataset.sh_degree)','gaussians = frozen_gaussians')
    body=body.replace('scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)',
                      'scene = frozen_scene')
    body=body.replace('os.path.join(dataset.model_path, "recon.ply")',
                      'os.path.join(output_path, "recon_layered.ply")')
    body=body.replace('os.path.join(dataset.model_path, "recon_post.ply")',
                      'os.path.join(output_path, "recon_post_layered.ply")')
    camera_index={id(c):i for i,c in enumerate(cameras)}
    for method in methods:
        dest=out/method;dest.mkdir(exist_ok=True)
        def cached_render(cam,*args):
            rgb,maps=frames[camera_index[id(cam)]]
            # Host channel selection is explicit for each method.
            return {'render':rgb,'expected_depth':maps[method],
                    'median_depth':maps[method]}
        ns=vars(host).copy()
        ns.update(frozen_gaussians=gaussians,frozen_scene=scene,
                  output_path=str(dest),render=cached_render)
        exec(compile(body,str(ROOT/'mesh_extract.py'),'exec'),ns)
        # Host mask is on CUDA, so cached input tensors use that device too.
        for j,(rgb,maps) in enumerate(frames):
            maps[method]=maps[method].cuda()
        t=time.time();ns['extract_mesh'](dataset,pipe,iteration,num_cluster)
        timings[method+'_fusion_seconds']=time.time()-t
        for _,maps in frames: maps[method]=maps[method].cpu()
        (out/('耗时'+suffix+'.json')).write_text(json.dumps(timings,indent=2))
        print('MESH_DONE',scan,method,flush=True)

if __name__=='__main__':
    parser=ArgumentParser();model=ModelParams(parser,sentinel=True);pipeline=PipelineParams(parser)
    parser.add_argument('--iteration',type=int,default=-1)
    parser.add_argument('--num_cluster',type=int,default=1)
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--methods',nargs='+',default=['median','gap_fixed','gap_adaptive','contribution_valley'])
    parser.add_argument('--check_only',action='store_true')
    args=get_combined_args(parser);safe_state(args.quiet)
    with torch.no_grad():run(model.extract(args),pipeline.extract(args),args.iteration,
                             args.methods,args.num_cluster,args.check_only)

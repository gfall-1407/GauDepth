"""Original camera alignment/evaluation; isolate mesh/metadata/result writes."""
import os,sys,inspect,subprocess,random
sys.dont_write_bytecode=True
from pathlib import Path
from argparse import ArgumentParser
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(ROOT),str(HERE.parent/'01_冻结模型分层')]
import torch,numpy as np
import evaluate_dtu_mesh as host
from arguments import ModelParams,PipelineParams,get_combined_args
from layered_depth import protect_scene_output

if __name__=='__main__':
    parser=ArgumentParser();model=ModelParams(parser,sentinel=True);PipelineParams(parser)
    parser.add_argument('--iteration',type=int,default=30000);parser.add_argument('--DTU',required=True);parser.add_argument('--method',required=True)
    args=get_combined_args(parser);random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.set_device(torch.device('cuda:0'))
    dataset=model.extract(args);output=HERE/'结果'/Path(dataset.source_path).name/args.method
    protect_scene_output(output)
    body=inspect.getsource(host.evaluate_mesh)
    for old,new in {
        'os.path.join(dataset.model_path, "recon_post.ply")':'os.path.join(output_path, "recon_post_surface_support.ply")',
        'os.path.join(dataset.model_path, "recon_aligned.ply")':'os.path.join(output_path, "recon_aligned_surface_support.ply")',
        'os.path.join(dataset.model_path, "vis")':'os.path.join(output_path, "vis")',
        'os.system(cmd)':'subprocess.run(cmd, shell=True, check=True)'}.items():
        assert body.count(old)==1,old;body=body.replace(old,new)
    ns=vars(host).copy();ns.update(args=args,output_path=str(output),subprocess=subprocess)
    os.chdir(ROOT);exec(compile(body,str(ROOT/'evaluate_dtu_mesh.py'),'exec'),ns);ns['evaluate_mesh'](dataset,args.iteration,args.DTU)
    (output/'vis'/'results.json').rename(output/'results_surface_support.json')
    meta=output/'multi_view_layered.json'
    if meta.exists():meta.rename(output/'multi_view_surface_support.json')
    # These are redundant evaluation copies inside this new method directory only.
    # Raw/post meshes and all metric JSON remain. Original evaluation itself is unchanged.
    (output/'recon_aligned_surface_support.ply').unlink()
    for p in (output/'vis').glob('*.ply'):p.unlink()

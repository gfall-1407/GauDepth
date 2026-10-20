"""Host camera alignment and untouched dtu_eval/eval.py, isolated mesh I/O."""
import os
import sys
sys.dont_write_bytecode=True
import inspect
import subprocess
from pathlib import Path
from argparse import ArgumentParser
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import torch
import random
import numpy as np
import evaluate_dtu_mesh as host
from arguments import ModelParams,PipelineParams,get_combined_args
from layered_depth import protect_scene_output

if __name__=='__main__':
    parser=ArgumentParser();model=ModelParams(parser,sentinel=True);PipelineParams(parser)
    parser.add_argument('--iteration',type=int,default=30000)
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--DTU',required=True)
    parser.add_argument('--method',required=True)
    args=get_combined_args(parser)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.cuda.set_device(torch.device('cuda:0'))
    dataset=model.extract(args)
    output=HERE/'结果'/Path(dataset.source_path).name/args.method
    protect_scene_output(output)
    body=inspect.getsource(host.evaluate_mesh)
    body=body.replace('os.path.join(dataset.model_path, "recon_post.ply")',
                      'os.path.join(output_path, "recon_post_layered.ply")')
    body=body.replace('os.path.join(dataset.model_path, "recon_aligned.ply")',
                      'os.path.join(output_path, "recon_aligned_layered.ply")')
    body=body.replace('os.path.join(dataset.model_path, "vis")',
                      'os.path.join(output_path, "vis")')
    body=body.replace('os.system(cmd)','subprocess.run(cmd, shell=True, check=True)')
    ns=vars(host).copy();ns.update(args=args,output_path=str(output),subprocess=subprocess)
    os.chdir(ROOT)
    exec(compile(body,str(ROOT/'evaluate_dtu_mesh.py'),'exec'),ns)
    ns['evaluate_mesh'](dataset,args.iteration,args.DTU)
    (output/'vis'/'results.json').rename(output/'results_layered.json')

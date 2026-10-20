"""Original extract_mesh body, reference support depth and isolated output paths."""
import sys
sys.dont_write_bytecode=True
import os,json,time,inspect
from pathlib import Path
from argparse import ArgumentParser
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mesh_extract as host
from arguments import ModelParams,PipelineParams,get_combined_args
from utils.general_utils import safe_state
from surface_support_depth import depth_maps,forward,reference_mesh_scene,protect_scene_output,exhaustive_check,method_names

def run(dataset,pipe,methods,lambdas,check_only):
    scan=Path(dataset.model_path).name;out=HERE/'结果'/scan;out.mkdir(parents=True,exist_ok=True)
    protect_scene_output(out)
    gaussians=host.GaussianModel(dataset.sh_degree)
    scene=host.Scene(dataset,gaussians,load_iteration=30000,shuffle=False)
    meta=out/'multi_view_layered.json'
    if meta.exists():meta.rename(out/'multi_view_surface_support.json')
    cameras=scene.getTrainCameras();frames=[];statistics=[];samples=[];started=time.time();mesh_scene=None
    params=dict(scan=scan,iteration=scene.loaded_iter,resolution=dataset.resolution,lambdas=lambdas,methods=methods,
        reference=str(Path(dataset.model_path)/'recon_post.ply'),voxel=.002,
        cameras=[dict(name=c.image_name,width=c.image_width,height=c.image_height) for c in cameras])
    key='_'.join(f'{l:.2f}' for l in lambdas)
    (out/f'运行参数_{key}.json').write_text(json.dumps(params,ensure_ascii=False,indent=2))
    if check_only:exhaustive_check()
    depth_dir=out/'深度';depth_dir.mkdir(exist_ok=True)
    needed=method_names(lambdas)
    previous_statistics={}
    required_suffixes={'l001' if lam==.01 else 'l003' for lam in lambdas}
    for p in out.glob('支持统计_*.json'):
        for st in json.loads(p.read_text()):
            if 'pixels' in st and required_suffixes.issubset(st.get('lambdas',{})):
                previous_statistics[st['view']]=st
    for i,cam in enumerate(cameras):
        file=depth_dir/(cam.image_name+'.npz');t=time.time()
        saved={}
        if file.exists():
            with np.load(file) as a:saved={k:a[k] for k in a.files}
        if all(m in saved for m in methods) and not check_only:
            if cam.image_name in previous_statistics:
                r,_,_=forward(cam,gaussians,pipe,torch.ones(3,device='cuda'),dataset.kernel_size)
                rgb=r[1].cpu();st=previous_statistics[cam.image_name].copy()
            else:
                # A stopped first pass may have valid depths but no final statistics JSON.
                # Reconstruct that missing statistic and verify, while retaining cached depths.
                if mesh_scene is None:mesh_scene=reference_mesh_scene(dataset.model_path)
                rgb,recomputed,st,_,_=depth_maps(cam,gaussians,pipe,dataset.kernel_size,mesh_scene,lambdas,validate=i==0)
                for m in methods:np.testing.assert_allclose(recomputed[m].numpy(),saved[m],rtol=2e-5,atol=2e-6)
                st['statistics_reconstructed']=True
            maps={m:torch.from_numpy(saved[m]) for m in methods};st['cached_depth']=True
        else:
            if mesh_scene is None:mesh_scene=reference_mesh_scene(dataset.model_path)
            rgb,maps,st,hit,picked=depth_maps(cam,gaussians,pipe,dataset.kernel_size,mesh_scene,lambdas,
                validate=i==0,capture=i==0 and scan in ['scan24','scan65','scan105','scan110','scan40','scan118'])
            if i==0:
                old=HERE.parent/'01_冻结模型分层'/'结果'/scan/'深度'/(cam.image_name+'.npz')
                with np.load(old) as a:
                    np.testing.assert_allclose(maps['expected'].numpy(),a['expected'],rtol=2e-5,atol=2e-6)
                    np.testing.assert_allclose(maps['host_median'].numpy(),a['median'],rtol=2e-5,atol=2e-6)
                if picked:
                    from PIL import Image
                    source=cam.original_image.cpu().numpy().transpose(1,2,0)
                    Image.fromarray((source*255).clip(0,255).astype(np.uint8)).save(out/f'{cam.image_name}_图像.png')
                    (out/f'{cam.image_name}_支持实例_{key}.json').write_text(json.dumps(picked,ensure_ascii=False,indent=2,allow_nan=False))
            if check_only:
                (out/'必要核对.json').write_text(json.dumps(dict(interval_check='40 cases pass',first_view=st),ensure_ascii=False,indent=2))
                print('CHECK_PASS',scan,st,flush=True);return
            saved.update({m:maps[m].numpy() for m in needed});saved['reference_hit']=hit
            np.savez_compressed(file,**saved)
            maps={m:maps[m] for m in methods}
        st['seconds']=time.time()-t;statistics.append(st);frames.append((rgb,maps))
        print('VIEW',scan,i+1,len(cameras),'cached' if st.get('cached_depth') else 'computed',flush=True)
    if statistics and all('pixels' in st for st in statistics):
        (out/f'支持统计_{key}.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2))
    timings=dict(depth_seconds=time.time()-started)
    body=inspect.getsource(host.extract_mesh)
    changes={
        'gaussians = GaussianModel(dataset.sh_degree)':'gaussians = frozen_gaussians',
        'scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)':'scene = frozen_scene',
        'os.path.join(dataset.model_path, "recon.ply")':'os.path.join(output_path, "recon_surface_support.ply")',
        'os.path.join(dataset.model_path, "recon_post.ply")':'os.path.join(output_path, "recon_post_surface_support.ply")'}
    for old,new in changes.items():assert body.count(old)==1,old;body=body.replace(old,new)
    index={id(c):i for i,c in enumerate(cameras)}
    for method in methods:
        dest=out/method;dest.mkdir(exist_ok=True)
        if (dest/'recon_post_surface_support.ply').exists():continue
        def cached_render(cam,*args):
            rgb,maps=frames[index[id(cam)]]
            return dict(render=rgb,expected_depth=maps[method],median_depth=maps[method])
        ns=vars(host).copy();ns.update(frozen_gaussians=gaussians,frozen_scene=scene,output_path=str(dest),render=cached_render)
        exec(compile(body,str(ROOT/'mesh_extract.py'),'exec'),ns)
        for _,maps in frames:maps[method]=maps[method].cuda()
        t=time.time();ns['extract_mesh'](dataset,pipe,30000,1)
        timings[method+'_fusion_seconds']=time.time()-t
        for _,maps in frames:maps[method]=maps[method].cpu()
        (out/f'耗时_{method}.json').write_text(json.dumps(timings,indent=2))
        print('MESH_DONE',scan,method,flush=True)

if __name__=='__main__':
    parser=ArgumentParser();model=ModelParams(parser,sentinel=True);pipeline=PipelineParams(parser)
    parser.add_argument('--methods',nargs='+',required=True);parser.add_argument('--lambdas',nargs='+',type=float,required=True)
    parser.add_argument('--check_only',action='store_true')
    args=get_combined_args(parser);safe_state(False)
    assert args.lambdas in [[.01],[.03],[.01,.03]]
    with torch.no_grad():run(model.extract(args),pipeline.extract(args),args.methods,args.lambdas,args.check_only)

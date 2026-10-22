"""Fuse the four saved means with host TSDF; compare against DTU in both directions."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
from depth_accuracy import SCANS,METHODS,load_scene,coordinate_check,write_csv
from sample import sha256,atomic_json,git


def evaluation_source():
    source=(ROOT/'dtu_eval/eval.py').read_text()
    edits={'np.random.default_rng()':'np.random.default_rng(20261022)',
           'mp.Pool()':'mp.Pool(processes=8)',
           'def write_vis_pcd(':'def unused_write_vis_pcd('}
    for old,new in edits.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source


def evaluate(mesh,scan,destination):
    # Execute the baseline evaluator in its own child. Only the seed, worker count,
    # visualization file writes and extra tail counters differ from its source.
    source=evaluation_source()
    argv=[str(ROOT/'dtu_eval/eval.py'),'--data',str(mesh),'--scan',str(scan),'--mode','mesh',
          '--dataset_dir','/root/autodl-tmp/DTU_MVS_Data','--vis_out_dir',str(destination)]
    payload='''import sys,json
from pathlib import Path
config=json.load(sys.stdin)
sys.argv=config['argv']
scope=globals()
scope.update(__file__=sys.argv[0],write_vis_pcd=lambda *a,**kw:None)
exec(compile(config['source'],sys.argv[0],'exec'),scope)
import numpy as np
tail={k:{'count':int(np.size(scope[k])),'at_least_20mm':int(np.count_nonzero(scope[k]>=20)),
         'untruncated_mean_mm':float(np.mean(scope[k]))} for k in ['dist_d2s','dist_s2d']}
Path(config['tail_path']).write_text(json.dumps(tail,indent=2))
'''
    with (destination/'evaluation.log').open('w') as log:
        subprocess.run([sys.executable,'-c',payload],input=json.dumps(dict(source=source,argv=argv,
            tail_path=str(destination/'evaluation_tail.json'))),text=True,stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=1200,cwd=ROOT)


def fuse(scan,means_root,out):
    import torch
    import open3d as o3d
    import mesh_extract as host
    from sample import forward
    from pixel_contribution import retained_mask
    assert os.uname().nodename=='autodl-container-d6a7459179-693ab6ee'
    assert not git('status','--short')
    inputs=means_root/f'scan{scan}';meta=json.loads((inputs/'metadata.json').read_text())
    assert meta['status']=='complete' and meta['original_model_unchanged']
    out.mkdir(parents=True,exist_ok=True)
    dataset,pipe,g,cameras,ply,digest=load_scene(scan,out)
    assert digest==meta['model_sha256'] and [c.image_name for c in cameras]==meta['camera_names']
    transform,alignment=coordinate_check(dataset,cameras)
    np.testing.assert_allclose(transform,meta['alignment']['transform'],rtol=0,atol=1e-6)
    state_path=out/'metadata.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(methods={})
    assert not state.get('git_commit') or state['git_commit']==git('rev-parse','HEAD')
    state.update(status='running',scan=scan,git_commit=git('rev-parse','HEAD'),mean_run_commit=meta['git_commit'],
        model_sha256=digest,alignment=alignment,methods=state['methods'],
        protocol='Host TSDF voxel .002/block16/count50000/depth_scale1/depth_max8/cluster1; original masks; shared finite-positive four-map coverage. Full means baseline recomputed, not historical score.',
        evaluator='Existing DTU evaluator, fixed default_rng seed20261022, eight workers; no visualization PLY; original ObsMask/Plane/.2mm sampling/20mm cutoff retained; extra untruncated means and excluded-tail counts.',
        host_mesh_source_sha256=sha256(ROOT/'mesh_extract.py'),evaluator_source_sha256=sha256(ROOT/'dtu_eval/eval.py'),
        camera_names=meta['camera_names'])
    atomic_json(state_path,state)
    required=[m for m in METHODS if m not in state['methods']]
    for m in METHODS:
        if m in state['methods']:
            assert sha256(out/m/'recon_post.ply')==state['methods'][m]['mesh_sha256']
            assert (out/m/'results.json').is_file() and (out/m/'evaluation_tail.json').is_file()
    if required:
        frames={};excluded=0;by_name={v['name']:v for v in meta['views']}
        white=torch.ones(3,dtype=torch.float32,device='cuda')
        with torch.no_grad():
            for cam in cameras:
                view=by_name[cam.image_name];assert sha256(inputs/view['file'])==view['sha256']
                with np.load(inputs/view['file']) as a:
                    index=a['pixel_index'];means=a['weighted_mean_depth']
                keep=retained_mask(cam.gt_mask.cpu().numpy(),(cam.image_height,cam.image_width))
                np.testing.assert_array_equal(index,np.flatnonzero(keep.ravel()))
                finite=np.isfinite(means).all(1)&(means>0).all(1);excluded+=int((~finite).sum())
                maps=np.zeros((4,cam.image_height*cam.image_width),np.float32)
                maps[:,index[finite]]=means[finite].T
                r,_,_=forward(cam,g,pipe,white,dataset.kernel_size)
                # All-member mean must still reproduce the host baseline on common support.
                torch.testing.assert_close(torch.from_numpy(means[finite,0]).cuda(),r[4].flatten()[torch.from_numpy(index[finite]).cuda().long()],rtol=2e-5,atol=2e-6)
                frames[cam.image_name]=(r[1].cpu(),maps.reshape(4,cam.image_height,cam.image_width))
                del r
        state['common_invalid_mask_pixels']=excluded;atomic_json(state_path,state)
        body=inspect.getsource(host.extract_mesh)
        edits={
            'gaussians = GaussianModel(dataset.sh_degree)':'gaussians = frozen_gaussians',
            'scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)':'scene = frozen_scene',
            'o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon.ply"), mesh.to_legacy())':'pass # do not archive the intermediate raw mesh',
            'os.path.join(dataset.model_path, "recon_post.ply")':'str(destination / "recon_post.ply")'}
        for old,new in edits.items():
            assert body.count(old)==1,old
            body=body.replace(old,new)
        for method in required:
            destination=out/method
            if destination.exists() and any(destination.iterdir()):raise FileExistsError(destination)
            destination.mkdir(exist_ok=True);started=time.time();j=METHODS.index(method)
            def cached_render(cam,*args):
                rgb,maps=frames[cam.image_name];d=torch.from_numpy(maps[j:j+1]).cuda()
                return dict(render=rgb,expected_depth=d,median_depth=d)
            scope=vars(host).copy();scope.update(frozen_gaussians=g,frozen_scene=SimpleNamespace(getTrainCameras=lambda:cameras),
                destination=destination,render=cached_render)
            exec(compile(body,str(ROOT/'mesh_extract.py'),'exec'),scope)
            scope['extract_mesh'](dataset,pipe,30000,1)
            mesh_path=destination/'recon_post.ply';mesh=o3d.io.read_triangle_mesh(str(mesh_path))
            assert len(mesh.vertices)>0 and len(mesh.triangles)>0
            mesh.transform(transform);aligned=destination/'evaluation_input_tmp.ply'
            assert not aligned.exists()
            assert o3d.io.write_triangle_mesh(str(aligned),mesh)
            evaluate(aligned,scan,destination)
            # This derived temporary is the only file removed; original data and post mesh stay.
            aligned_hash=sha256(aligned);aligned.unlink()
            result=json.loads((destination/'results.json').read_text())
            assert all(np.isfinite(result[k]) for k in ('mean_d2s','mean_s2d','overall'))
            state['methods'][method]=dict(**result,mesh_sha256=sha256(mesh_path),aligned_temporary_sha256=aligned_hash,
                seconds=time.time()-started,tail=json.loads((destination/'evaluation_tail.json').read_text()))
            atomic_json(state_path,state);print('MESH',scan,method,result,flush=True)
            if (out.parent/'stop_after_stage').exists():break
    assert sha256(ply)==digest
    state.update(status='complete' if len(state['methods'])==4 else 'partial',original_model_unchanged=True)
    atomic_json(state_path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scan',type=int,choices=SCANS)
    p.add_argument('--all-scans',action='store_true');p.add_argument('--means-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if bool(a.scan)==bool(a.all_scans):p.error('choose scan or all-scans')
    if a.all_scans:
        a.output.mkdir(parents=True,exist_ok=True)
        for scan in SCANS:
            d=a.output/f'scan{scan}';d.mkdir(exist_ok=True)
            with (d/'run.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--scan',str(scan),'--means-root',str(a.means_root),
                    '--output',str(d)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400,cwd=ROOT)
            rows=[]
            for done in SCANS:
                m=a.output/f'scan{done}/metadata.json'
                if m.exists():
                    state=json.loads(m.read_text())
                    rows += [dict(scan=done,method=k,**{s:v[s] for s in ('mean_d2s','mean_s2d','overall','seconds')}) for k,v in state['methods'].items()]
            write_csv(a.output/'全部scan重建精度.csv',rows)
            if (a.output/'stop_after_stage').exists():break
    else:fuse(a.scan,a.means_root,a.output)

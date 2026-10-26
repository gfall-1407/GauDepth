"""Matched PLY warm-start adaptation with a static gap teacher, never physical GT."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import numpy as np
from photo_surface_probe import HERE, REPO, sha256, save, source_info
sys.path.insert(0, str(REPO))


def gap_loss(depth, teacher, gate, mask, h_scene):
    import torch
    import torch.nn.functional as F
    if depth.shape != teacher.shape or gate.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("Unaligned depth/teacher/gate/mask")
    if not math.isfinite(h_scene) or h_scene <= 0 or not bool(mask.any()):
        raise ValueError("Positive physical bandwidth and foreground required")
    if not bool(torch.isfinite(depth).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("Nonfinite depth/teacher cannot be silently zeroed")
    residual = (depth[mask]-teacher.detach()[mask])/h_scene
    return (F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1., reduction="none") * gate[mask].detach()).sum()/mask.sum()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def apply_loss_backward(base, weighted_gap, parameters, route, verify=False):
    """Route only the added term; the full baseline still updates every leaf."""
    import torch
    if route == 'full':
        (base+weighted_gap).backward()
        return None
    if route != 'xyz': raise ValueError('Unknown gradient route')
    base.backward(retain_graph=True)
    before = {k:None if p.grad is None else p.grad.detach().clone()
              for k,p in parameters.items()} if verify else None
    xyz=parameters['xyz']
    extra=torch.autograd.grad(weighted_gap,xyz,allow_unused=True)[0]
    if extra is None or not bool(torch.isfinite(extra).all()):
        raise ValueError('Added xyz gradient missing or nonfinite')
    if xyz.grad is None: xyz.grad=extra.detach().clone()
    else: xyz.grad.add_(extra)
    if not verify: return None
    for name,param in parameters.items():
        if name=='xyz': continue
        previous=before[name]
        if (previous is None) != (param.grad is None) or (previous is not None and not torch.equal(previous,param.grad)):
            raise ValueError('Added term wrote a forbidden parameter gradient: '+name)
    expected=extra if before['xyz'] is None else before['xyz']+extra
    if not torch.equal(xyz.grad,expected): raise ValueError('xyz gradient sum differs')
    b=before['xyz'];bn=0. if b is None else float(b.norm());en=float(extra.norm())
    cosine=None if b is None or bn==0 or en==0 else float((b*extra).sum())/(bn*en)
    return dict(status='PASS_exact_direct_route',forbidden=('scaling','rotation','opacity'),
        forbidden_gradient_bitwise_unchanged=True,xyz_addition_bitwise_equal=True,
        baseline_xyz_norm=bn,weighted_extra_xyz_norm=en,baseline_extra_xyz_cosine=cosine,
        limitation='Only the direct added gradient is restricted; later shared-geometry and baseline gradients may change. Not proof of physical displacement.')


def check():
    import torch
    d = torch.tensor([2., 2.], requires_grad=True)
    t = torch.tensor([1., 1.], requires_grad=True)
    mask = torch.ones(2, dtype=torch.bool)
    loss = gap_loss(d, t, torch.tensor([True, False]), mask, 1.)
    loss.backward()
    assert d.grad[0] > 0 and d.grad[1] == 0 and t.grad is None
    d.grad = None
    (0.*gap_loss(d, t, mask, mask, 1.)).backward()
    assert torch.equal(d.grad, torch.zeros_like(d))
    assert gap_loss(d.detach()-.01, t, mask, mask, 1.) < gap_loss(d.detach(), t, mask, mask, 1.)
    for route in ('full','xyz'):
        parameters={k:torch.tensor(float(i+1),requires_grad=True) for i,k in enumerate(('xyz','scaling','rotation','opacity'))}
        x,s,r,o=parameters.values();base=(x+s+r+o)**2;term=.25*(x*s+r*o)**2
        bg=torch.autograd.grad(base,list(parameters.values()),retain_graph=True)
        tg=torch.autograd.grad(term,list(parameters.values()),retain_graph=True)
        qa=apply_loss_backward(base,term,parameters,route,verify=route=='xyz')
        for i,param in enumerate(parameters.values()):
            expected=bg[i]+tg[i] if route=='full' or i==0 else bg[i]
            assert torch.equal(param.grad,expected)
        if route=='xyz':assert qa['forbidden_gradient_bitwise_unchanged']
    print("CPU smooth-L1 direction/gate-zero/teacher-detach/lambda-zero: PASS")
    print("CPU full baseline retained; added term full/xyz route and forbidden gradients: PASS")


def run(args):
    source = source_info()
    if socket.gethostname() != "autodl-container-5d5147b55f-67549ee6":
        raise ValueError("Only explicitly authorized G0; independent guard required externally")
    out = args.output.resolve()
    if not args.output.is_absolute() or out.exists():
        raise ValueError("New absolute output required")
    for root in (args.model_root, args.source_root, args.teacher_root):
        root = root.resolve()
        if root == out or root in out.parents or out in root.parents:
            raise ValueError("Output and protected inputs must be independent")
    from dense_huber_mesh import cfg_values, verify_camera_reference, HOST_SHA
    refpath = HERE/'G1_Huber全场景输入参考.json'
    ref = json.loads(refpath.read_text(encoding='utf-8'))['scans'][str(args.scan)]
    model = args.model_root/f'scan{args.scan}'
    ply, cfg = model/'point_cloud/iteration_30000/point_cloud.ply', model/'cfg_args'
    if sha256(ply) != ref['model_sha256'] or sha256(cfg) != ref['cfg_sha256']:
        raise ValueError("Canonical model/cfg mismatch")
    for rel, digest in ref['source_inputs'].items():
        if sha256(args.source_root/rel) != digest: raise ValueError("Canonical camera input mismatch: "+rel)
    meta_path = args.teacher_root/'metadata.json'
    teacher_meta_sha = sha256(meta_path)
    tm = json.loads(meta_path.read_text(encoding='utf-8'))
    if (tm['status'] != 'cache_complete' or tm['scan'] != args.scan or tm['h_mm'] != 1. or
        tm['model_sha256'] != ref['model_sha256'] or tm['cfg_sha256'] != ref['cfg_sha256']):
        raise ValueError("Teacher is not the completed canonical fixed1mm gap cache")
    if tm['source']['commit'] != args.teacher_commit or 'gap_huber' not in tm['methods']:
        raise ValueError("Teacher source/method differs from the explicitly pinned cache")
    if [v['name'] for v in tm['views']] != [v['name'] for v in ref['views']]:
        raise ValueError("Teacher camera order/inventory mismatch")
    for v in tm['views']:
        path = (args.teacher_root/v['file']).resolve()
        if args.teacher_root.resolve() not in path.parents or sha256(path) != v['sha256']:
            raise ValueError("Teacher frame path/SHA mismatch")
    config = cfg_values(cfg)
    if float(config['depth_ratio']) != 0. or config.get('eval',False) or config['use_decoupled_appearance'] != 3:
        raise ValueError("Expected G1 ratio0/PGSR3/training-camera configuration")
    out.mkdir(parents=True)
    stop = {'requested':False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    state = dict(status='initializing', source=source, args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_model_sha256=ref['model_sha256'], source_cfg_sha256=ref['cfg_sha256'],
        reference_sha256=sha256(refpath), teacher_metadata_sha256=teacher_meta_sha,
        teacher_source=tm['source'], teacher_views=tm['views'], actual_updates=0,
        interpretation='Matched PLY warm-start adaptation; Adam restarted, PGSR appearance freshly initialized with same seed. Original optimizer/appearance state unavailable; not exact resume. Teacher is frozen readout, not GT. No densify/reset/filter recompute/SH increment.',
        optimizer_provenance='Explicit current OptimizationParams defaults; original training optimizer configuration unavailable.')
    save(out/'metadata.json',state)
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from diff_gaussian_rasterization import _C
    from sample import protect_scene
    from depth_accuracy import coordinate_check
    from pixel_contribution import retained_mask
    from utils.graphics_utils import depth_to_normal
    from utils.loss_utils import L1_loss_appearance, ssim, PatchMatch
    if sha256(_C.__file__) != HOST_SHA: raise ValueError('Original CUDA rasterizer provenance mismatch')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    p=argparse.ArgumentParser();mp=ModelParams(p);op=OptimizationParams(p);pp=PipelineParams(p)
    values=vars(p.parse_args([]));values.update(config)
    values.update(model_path=str(model),source_path=str(args.source_root/f'scan{args.scan}'))
    ns=argparse.Namespace(**values);dataset=mp.extract(ns);pipe=pp.extract(ns)
    # Optimization defaults are intentionally independent of incomplete cfg_args.
    opt=op.extract(p.parse_args([]))
    opt.regularization_from_iter=30001
    g=GaussianModel(dataset.sh_degree)
    with protect_scene(out): scene=Scene(dataset,g,load_iteration=30000,shuffle=False)
    cameras=scene.getTrainCameras()
    verify_camera_reference(cameras,ref)
    _,alignment=coordinate_check(dataset,cameras)
    scale=float(alignment['scale_to_dtu_mm']);h_scene=1./scale
    np.testing.assert_allclose(scale,tm['alignment']['scale_to_dtu_mm'],rtol=0,atol=1e-12)
    g.spatial_lr_scale=float(scene.cameras_extent)
    if not math.isfinite(g.spatial_lr_scale) or g.spatial_lr_scale<=0:raise ValueError('Invalid nonzero position LR scale')
    scene.model_path=dataset.model_path=str(out)
    g.training_setup(opt)
    torch.cuda.reset_peak_memory_stats()
    (out/'cfg_args').write_text(str(argparse.Namespace(**vars(dataset))),encoding='utf-8')
    parameters={name:getattr(g,'_'+name) for name in ('xyz','scaling','rotation','opacity')}
    tracked=dict(parameters,features_dc=g._features_dc,features_rest=g._features_rest,filter_3D=g.filter_3D,appearance=g._appearance_embeddings)
    initial={k:tensor_sha(v) for k,v in tracked.items() if v is not None}
    count=args.smoke_steps or args.steps
    rng=random.Random(args.seed);stack=[];schedule=[]
    for _ in range(count):
        if not stack:stack=list(range(len(cameras)))
        ci=stack.pop(rng.randrange(len(stack)));near=cameras[ci].nearest_id
        schedule.append((ci,int(rng.choice(near)) if len(near) else None))
    state.update(status='running',options=vars(opt),dataset=vars(dataset),alignment=alignment, h_scene=h_scene,
        initial_tensor_sha256=initial,camera_schedule=schedule,
        schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        spatial_lr_scale=g.spatial_lr_scale,original_C_path=_C.__file__,original_C_sha256=sha256(_C.__file__),
        cuda=torch.version.cuda,torch=torch.__version__,seed=args.seed,
        original_C_source_limitation='Installed f97d binary preserved; original source/build gap not resolved by matching Git.',
        source_sha256={name:sha256(REPO/name) for name in ('train.py','scene/gaussian_model.py','utils/loss_utils.py','gaussian_renderer/__init__.py')})
    save(out/'metadata.json',state)
    patchmatch=PatchMatch(opt.multi_view_patch_size,opt.multi_view_pixel_noise_th,dataset.kernel_size,pipe,debug=False,model_path=str(out))
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device='cuda')
    started=time.monotonic();qa_done=False;completed=False
    def persist(status):
        iteration=30000+state['actual_updates']
        dest=out/f'point_cloud/iteration_{iteration}/point_cloud.ply'
        g.save_ply(str(dest))
        checkpoint=out/'last_checkpoint.pth';temp=out/'last_checkpoint.pth.tmp'
        torch.save(dict(capture=g.capture(),filter_3D=g.filter_3D.detach(),iteration=iteration,
            source=source,teacher_metadata_sha256=teacher_meta_sha,options=vars(opt),gap_gradient_route=args.gap_gradient_route,
            random_state=random.getstate(),numpy_state=np.random.get_state(),torch_state=torch.get_rng_state(),
            cuda_state=torch.cuda.get_rng_state_all(),schedule=schedule),temp)
        os.replace(temp,checkpoint)
        if sha256(ply)!=ref['model_sha256'] or sha256(cfg)!=ref['cfg_sha256'] or sha256(meta_path)!=teacher_meta_sha:
            raise ValueError('Protected original/teacher metadata changed')
        state.update(status=status,iteration=iteration,checkpoint_sha256=sha256(checkpoint),ply_sha256=sha256(dest),
            final_tensor_sha256={k:tensor_sha(v) for k,v in tracked.items() if v is not None},seconds=time.monotonic()-started)
        state.update(gpu=torch.cuda.get_device_name(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        state['changed_tensors']={k:state['final_tensor_sha256'][k]!=v for k,v in initial.items()}
        save(out/'metadata.json',state)
    try:
        with (out/'loss.jsonl').open('x',encoding='utf-8') as log:
            for local,(ci,ni) in enumerate(schedule,1):
                if stop['requested'] or (out/'stop_after_step').exists() or time.monotonic()-started>=args.max_seconds:break
                # Step-specific seed makes stochastic renderer/PatchMatch choices arm-matched.
                random.seed(args.seed+local);np.random.seed(args.seed+local);torch.manual_seed(args.seed+local);torch.cuda.manual_seed_all(args.seed+local)
                camera=cameras[ci];v=tm['views'][ci];frame_path=args.teacher_root/v['file']
                if sha256(frame_path)!=v['sha256']:raise ValueError('Teacher changed after preflight')
                with np.load(frame_path) as frame:
                    mask_np=retained_mask(camera.gt_mask.cpu().numpy(),(camera.image_height,camera.image_width))
                    if not np.array_equal(mask_np,frame['gt_mask'].astype(bool)):raise ValueError('Teacher/camera mask mismatch')
                    target_np=frame['gap_huber'];expected_np=frame['expected']
                    if target_np.shape!=expected_np.shape or not np.isfinite(target_np).all() or not np.isfinite(expected_np).all():raise ValueError('Invalid teacher shapes/values')
                    target=torch.from_numpy(target_np.copy()).cuda();mask=torch.from_numpy(mask_np).cuda().reshape(target.shape)
                    gate=torch.from_numpy((target_np!=expected_np).copy()).cuda() & mask
                lr=g.update_learning_rate(30000+local)
                xyz_lr=next(x['lr'] for x in g.optimizer.param_groups if x['name']=='xyz')
                if not math.isfinite(xyz_lr) or xyz_lr<=0:raise ValueError('Zero/nonfinite position LR')
                pkg=render(camera,g,pipe,bg,dataset.kernel_size,require_depth=True)
                D=pkg['expected_depth'];normal=depth_to_normal(camera,D)
                if local==1:
                    current=D.detach().cpu().numpy()
                    if not np.array_equal(current,expected_np,equal_nan=True):
                        raise ValueError('Warm-start native expected differs from the pinned teacher initial expected')
                    state['initial_expected_teacher_bitwise']='PASS'
                rgb=(1-opt.lambda_dssim)*L1_loss_appearance(pkg['render'],camera.original_image.cuda(),g,camera.uid)+opt.lambda_dssim*(1-ssim(pkg['render'].unsqueeze(0),camera.original_image.cuda().unsqueeze(0)))
                norm=(1-torch.linalg.vecdot(pkg['normal'],normal,dim=0)).mean()
                ncc,geo=patchmatch(g,pkg,camera,cameras[ni] if ni is not None else None,30000+local,normal)
                gap=gap_loss(D,target,gate,mask,h_scene)
                base=rgb+opt.lambda_depth_normal*norm+opt.lambda_multi_view_ncc*ncc+opt.lambda_multi_view_geo*geo
                if not qa_done and bool(gate.any()):
                    names=['depth']+list(parameters)
                    grads=torch.autograd.grad(gap,[D]+list(parameters.values()),retain_graph=True,allow_unused=True)
                    def grad_info(x):return None if x is None else dict(finite=bool(torch.isfinite(x).all()),norm=float(x.detach().norm()))
                    qa={k:grad_info(x) for k,x in zip(names,grads)}
                    basegrads=torch.autograd.grad(base,list(parameters.values()),retain_graph=True,allow_unused=True)
                    detached=D.detach().clone().requires_grad_(True);probe=gap_loss(detached,target,gate,mask,h_scene)
                    dg=torch.autograd.grad(probe,detached)[0];eps=h_scene*.001
                    moved=detached.detach()-eps*dg/(dg.norm()+1e-30)
                    qa['direction_before']=float(probe.detach());qa['direction_after']=float(gap_loss(moved,target,gate,mask,h_scene))
                    state['gap_gradient_qa']=qa;state['baseline_geometry_gradients']={k:grad_info(x) for k,x in zip(parameters,basegrads)}
                    if qa['depth'] is None or not qa['depth']['finite'] or qa['depth']['norm']<=0 or not any(qa[k] and qa[k]['finite'] and qa[k]['norm']>0 for k in parameters) or qa['direction_after']>=qa['direction_before']:
                        raise ValueError('New gap term lacks verified finite geometry gradient/direction')
                    qa_done=True;save(out/'metadata.json',state)
                loss=base+args.lambda_gap*gap
                if not bool(torch.isfinite(loss).all()):raise ValueError('Nonfinite training loss')
                route_qa=apply_loss_backward(base,args.lambda_gap*gap,parameters,args.gap_gradient_route,
                    verify=args.gap_gradient_route=='xyz' and 'gradient_route_qa' not in state and bool(gate.any()))
                if route_qa is not None:
                    state['gradient_route_qa']=route_qa;save(out/'metadata.json',state)
                for name,param in parameters.items():
                    if param.grad is not None and not bool(torch.isfinite(param.grad).all()):raise ValueError('Nonfinite '+name+' gradient')
                g.optimizer.step();g.optimizer.zero_grad(set_to_none=True)
                state['actual_updates']=local
                if local==1 or local%100==0 or local==count:
                    row=dict(step=local,iteration=30000+local,camera=camera.image_name,neighbour=ni,xyz_lr=xyz_lr,
                        rgb=float(rgb.detach()),normal=float(norm.detach()),ncc=float(ncc.detach()),geo=float(geo.detach()),
                        gap_unweighted=float(gap.detach()),gap_weighted=float((args.lambda_gap*gap).detach()),gap_pixels=int(gate.sum()),foreground_pixels=int(mask.sum()),gap_gradient_route=args.gap_gradient_route)
                    log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno());state['last_loss']=row;save(out/'metadata.json',state)
            completed=state['actual_updates']==count
            # A completed smoke must demonstrate real updates, including position.
            if completed and args.smoke_steps:
                state['smoke_parameter_changed']={k:tensor_sha(v)!=initial[k] for k,v in parameters.items()}
                if not state['smoke_parameter_changed']['xyz'] or not any(state['smoke_parameter_changed'].values()):
                    raise ValueError('Smoke did not update geometry/position despite positive position LR')
            if completed and not qa_done:raise ValueError('No nonempty gap gradient QA during completed adaptation')
        persist('smoke_complete' if completed and args.smoke_steps else ('complete' if completed else 'partial_stopped'))
    except BaseException as exc:
        state['error']=repr(exc)
        persist('failed_saved')
        raise
    return state


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,default=Path('/root/GauDepth_HuberRepair_20261026/inputs_G1/outtest'))
    p.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/DTU'))
    p.add_argument('--teacher-root',type=Path,default=Path('/root/GauDepth_Gap_20261027/scan40_v1/cache40'))
    from night_depth_batch import FIXED
    p.add_argument('--output',type=Path);p.add_argument('--scan',type=int,choices=tuple(FIXED),default=40)
    p.add_argument('--teacher-commit',default='db7dbf75ef1e473478249c2cff1b5a844ba2eea6',help='Exact authorized source commit recorded in completed teacher metadata')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--lambda-gap',type=float,default=.01);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--gap-gradient-route',choices=('full','xyz'),default='full')
    p.add_argument('--max-seconds',type=float,default=900);p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if not 1<=a.steps<=2000 or not 0<=a.smoke_steps<=2000 or not math.isfinite(a.lambda_gap) or a.lambda_gap<0 or not math.isfinite(a.max_seconds) or a.max_seconds<=0:p.error('Invalid bounded steps/lambda/time')
    if a.check:source_info();check()
    elif a.output is None:p.error('--output required')
    else:print(json.dumps(run(a),ensure_ascii=False))


if __name__=='__main__':cli()

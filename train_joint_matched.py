"""Fixed-pool Expected/joint continuation with fresh Adam and counted updates.

Two commands: prepare freezes track anchors plus the shared update schedule;
train executes one arm in a fresh output directory. No original PLY is modified.
Require an external timeout in addition to the internal budget. A failed root,
support budget or refinement check aborts; it never silently removes supervision.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import torch
from arguments import OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.loss_utils import L1_loss_appearance,ssim
from joint_tv_inputs import MODEL_IDENTITIES,verify_inputs,sha256
from joint_tv_model import FIELD_PROTOCOL,candidate_depths,rays_for_pixels
from joint_training_anchors import load_cameras,candidate_track_groups,sample_map,make_schedule

ROOT=Path('/root/joint_tv_runs')
PARAMETER_NAMES=('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation')
GEOMETRY_PROTOCOL=dict(near=.2001,far=8.,opacity_factor=1-1e-6,
    field='joint TV of full covariance Gaussian potential',
    support='fresh float64 raw-parameter profiles each call; fixed during backward',
    selection_change_from_old_probe='old probe selected members with float32 profiles',
    refinement='4/8; unresolved rays checked8/16; no fallback depth',
    no_root_policy='abort fixed-pool training, never drop/resample supervision',
    completeness_certified=False)


class GuardFailure(RuntimeError):
    def __init__(self,message,diagnostic):
        super().__init__(message);self.diagnostic=diagnostic


def json_safe(value):
    if isinstance(value,torch.Tensor):return json_safe(value.detach().cpu().tolist())
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked_source(expected_head):
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=expected_head:raise ValueError('unexpected source commit')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('experiment checkout must be clean')
    return head


def create_output(path):
    root=ROOT.resolve();out=Path(path).resolve()
    if not out.is_relative_to(root) or out==root:raise ValueError('dedicated system output directory required')
    root.mkdir(parents=True,exist_ok=True)
    if root.stat().st_dev!=Path('/root').stat().st_dev:raise ValueError('output root is not system disk')
    if shutil.disk_usage(root).free<2<<30:raise ValueError('at least2GiB system free space required')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False)
    return out


def native_config(path):
    """Parse native Namespace literals, never eval configuration code."""
    path=Path(path).resolve();expr=ast.parse(path.read_text(),mode='eval').body
    if not isinstance(expr,ast.Call) or not isinstance(expr.func,ast.Name) or expr.func.id!='Namespace' or expr.args:
        raise ValueError('native config must be literal Namespace keyword arguments')
    values={kw.arg:ast.literal_eval(kw.value) for kw in expr.keywords if kw.arg is not None}
    if any(kw.arg is None for kw in expr.keywords):raise ValueError('config unpacking is unsupported')
    # PLY omits appearance state. PGSR is supported with identical fresh zero
    # coefficients in both arms, never described as recovered original state.
    app=values.get('use_decoupled_appearance',0)
    if app not in (0,3):raise ValueError('this PLY-only pilot supports NO(0) and fresh PGSR(3)')
    for key in ('white_background','kernel_size','sh_degree'):
        if key not in values:raise ValueError('missing native renderer setting '+key)
    selected=dict(white_background=bool(values['white_background']),kernel_size=float(values['kernel_size']),
                  sh_degree=int(values['sh_degree']),use_decoupled_appearance=int(app),eval=bool(values.get('eval',False)))
    if values.get('images','images') not in ('images',None):raise ValueError('native images directory must be images')
    if not 0<=selected['kernel_size']<=1 or not 0<=selected['sh_degree']<=3:
        raise ValueError('unsupported native renderer settings')
    return dict(path=str(path),sha256=sha256(path),settings=selected)


def runtime_identity():
    wrapper=importlib.import_module('diff_gaussian_rasterization')
    extension=importlib.import_module('diff_gaussian_rasterization._C')
    return dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),rasterizer_wrapper_path=wrapper.__file__,
        rasterizer_wrapper_sha256=sha256(wrapper.__file__),rasterizer_binary_path=extension.__file__,
        rasterizer_binary_sha256=sha256(extension.__file__),
        extension_source_identity='binary identity recorded; compile-source provenance must be independently checked')


def seed_all(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)


def load_model(identity,settings,scene_radius,camera_count,training=False,optimizer_settings=None):
    g=GaussianModel(settings['sh_degree']);g.load_ply(identity['input_ply'])
    if identity['expected_rows'] is not None and len(g.get_xyz)!=identity['expected_rows']:
        raise ValueError('original model row count mismatch')
    g.spatial_lr_scale=float(scene_radius);g.create_app_model(camera_count,settings['use_decoupled_appearance'])
    # PLY lacks training accumulators. Fixed topology does not consume radii,
    # but store a correctly sized fresh buffer in archival model captures.
    g.max_radii2D=torch.zeros(len(g.get_xyz),device=g.get_xyz.device,dtype=g.get_xyz.dtype)
    if training:
        opt=SimpleNamespace(**optimizer_settings);g.training_setup(opt)
        if len(g.optimizer.state)!=0:raise AssertionError('Adam must start empty')
    return g


def model_parameter_names(g):
    return PARAMETER_NAMES+(('_appearance_embeddings',) if g._appearance_embeddings is not None else ())


def model_fingerprint(g):
    h=hashlib.sha256()
    for name in model_parameter_names(g)+('filter_3D',):
        value=getattr(g,name).detach().contiguous().cpu().numpy()
        h.update(name.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    h.update(str(g.active_sh_degree).encode())
    h.update(str(g.app_model.value).encode())
    return h.hexdigest()


def renderer_settings(settings):
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    background=torch.tensor([1.,1.,1.] if settings['white_background'] else [0.,0.,0.],device='cuda')
    return pipe,background,settings['kernel_size']


def evaluate_pairs(g,cameras,group,anchor_indices,settings,protocol,*,joint_grad=False,native_grad=False,native_packages=None):
    """Same availability check in both arms; only loss readout differs."""
    selected=[group['tracks'][j] for j in anchor_indices];count=len(selected)
    origins=[];directions=[];expected=[];alpha=[];packages=[]
    pipe,background,kernel=renderer_settings(settings)
    for side,cam_index in enumerate(group['camera_indices']):
        cam=cameras[cam_index]
        xy=torch.tensor([t['xy'][side] for t in selected],device='cuda',dtype=torch.float64)
        o,d=rays_for_pixels(cam,xy);origins.append(o);directions.append(d)
        with torch.set_grad_enabled(native_grad):
            if native_packages is not None:
                if native_grad:raise ValueError('native graph cache is only for immutable no-grad checks')
                pkg=native_packages[side]
            else:
                pkg=render(cam,g,pipe,background,kernel,require_depth=True)
            expected.append(sample_map(pkg['expected_depth'],xy))
            alpha.append(sample_map(pkg['mask'],xy))
        packages.append(pkg)
    origins=torch.cat(origins);directions=torch.cat(directions)
    dj,vj,diag=candidate_depths(g,origins,directions,epsilon=protocol['member_epsilon'],
        max_members=protocol['max_members'],verify=True,require_grad=joint_grad)
    de=torch.cat(expected);aa=torch.cat(alpha)
    ve=torch.isfinite(de)&(de>protocol['near'])&(de<protocol['far'])
    ve=ve&torch.isfinite(aa)&(aa>=protocol['min_native_alpha'])
    valid=vj&ve
    return dict(joint=dj,expected=de,origins=origins,directions=directions,
        joint_valid=vj,native_valid=ve,valid=valid,native_alpha=aa,diagnostic=diag,
        packages=packages,count=count)


def require_available(ev,group,stage):
    if not bool(ev['valid'].all()):
        raise GuardFailure('fixed anchor unavailable at '+stage,
            dict(stage=stage,camera_indices=group['camera_indices'],joint_valid=ev['joint_valid'],
                 native_valid=ev['native_valid'],native_alpha=ev['native_alpha'],candidate=ev['diagnostic']))


def full_pool_check(g,cameras,experiment,deadline):
    n=0
    for group in experiment['groups']:
        cached=None
        for first in range(0,len(group['tracks']),experiment['pairs_per_step']):
            deadline();ii=list(range(first,min(first+experiment['pairs_per_step'],len(group['tracks']))))
            ev=evaluate_pairs(g,cameras,group,ii,experiment['native_config']['settings'],
                              experiment['geometry_protocol'],native_packages=cached)
            cached=ev['packages']
            require_available(ev,group,'full_pool');n+=len(ii)
    return n


def optimizer_defaults():
    return vars(OptimizationParams(argparse.ArgumentParser(add_help=False)))


def prepare(args):
    head=checked_source(args.expected_head);out=create_output(args.output);started=time.monotonic()
    def deadline():
        if time.monotonic()-started>args.max_seconds:raise TimeoutError('preparation budget exhausted')
    identity=verify_inputs(args.input_ply,args.source_path,args.model_id)
    config=native_config(args.native_config);seed_all(args.seed)
    protocol=dict(GEOMETRY_PROTOCOL,member_epsilon=args.member_epsilon,max_members=args.max_members,
                  min_native_alpha=args.min_native_alpha,max_initial_pair_gap=args.max_initial_pair_gap)
    field_protocol=copy.deepcopy(FIELD_PROTOCOL)
    if any(protocol[k]!=field_protocol[k] for k in ('near','far','opacity_factor')):
        raise ValueError('geometry declaration differs from shared field protocol')
    manifest=dict(status='preparing',head=head,input_identity=identity,native_config=config,
        geometry_protocol=protocol,field_protocol=field_protocol,resolution=args.resolution,
        seed=args.seed,pairs_per_step=args.pairs_per_step,
        updates=args.updates,full_pool_every=args.full_pool_every,actual_optimizer_steps=0,
        runtime_identity=runtime_identity(),schema='fixed_colmap_joint_expected_v1',
        appearance_initialization=('fresh zero PGSR coefficients in both arms; original appearance state not restored'
            if config['settings']['use_decoupled_appearance']==3 else 'native appearance NO'))
    write_json(out/'prepare_status.json',manifest)
    try:
        cameras,metadata=load_cameras(args.source_path,args.resolution,'cpu',train_only=True,
                                     eval_split=config['settings']['eval'])
        g=load_model(identity,config['settings'],metadata['scene_radius'],len(cameras))
        candidates=candidate_track_groups(args.source_path,metadata,neighbors=args.neighbors,
            max_groups=args.max_groups,candidate_limit=args.candidate_limit,seed=args.seed)
        groups=[];selection=[]
        for group in candidates:
            kept=[];tested=0;reason_counts={'unavailable':0,'initial_pair_gap':0};cached=None
            for start in range(0,len(group['tracks']),args.pairs_per_step):
                deadline();ii=list(range(start,min(start+args.pairs_per_step,len(group['tracks']))))
                ev=evaluate_pairs(g,cameras,group,ii,config['settings'],protocol,native_packages=cached)
                cached=ev['packages']
                k=len(ii);good=ev['valid'].reshape(2,k).all(0)
                gaps=[]
                for key in ('joint','expected'):
                    x=ev['origins']+ev['directions']*ev[key][:,None]
                    gaps.append(torch.linalg.vector_norm(x[:k]-x[k:],dim=-1)/metadata['scene_radius'])
                gap=torch.maximum(*gaps);gap_ok=torch.isfinite(gap)&(gap<=args.max_initial_pair_gap)
                reason_counts['unavailable']+=int((~good).sum())
                reason_counts['initial_pair_gap']+=int((good&(~gap_ok)).sum())
                tested+=k
                for offset in (good&gap_ok).nonzero().flatten().tolist():
                    kept.append(group['tracks'][ii[offset]])
                if len(kept)>=args.anchors_per_group:break
            selection.append(dict(camera_indices=group['camera_indices'],tested=tested,qualified=len(kept),
                                  accepted=min(len(kept),args.anchors_per_group) if len(kept)>=args.pairs_per_step else 0,
                                  rejected=reason_counts))
            if len(kept)>=args.pairs_per_step:
                groups.append(dict(camera_indices=group['camera_indices'],camera_names=group['camera_names'],
                                   tracks=kept[:args.anchors_per_group]))
            print(json.dumps(selection[-1]),flush=True)
        if len(groups)<args.min_groups:raise RuntimeError('insufficient fixed anchor groups; do not silently relax selection')
        schedule=make_schedule(groups,args.updates,args.pairs_per_step,args.seed+1)
        manifest.update(status='complete',camera_metadata=metadata,groups=groups,schedule=schedule,
            optimizer_settings=optimizer_defaults(),base_iteration=7000,steps_are='fresh Adam updates from original7k PLY',
            fixed_topology=True,fixed_3d_filter=True,native_rgb_loss='(1-lambda_dssim)*native_L1+lambda_dssim*(1-SSIM)',
            active_sh_degree_fixed=True,native_depth_normal_loss_enabled=False,native_patchmatch_enabled=False,
            opacity_reset_enabled=False,full_original_training_recipe_reproduced=False,
            geometry_loss='mean L2 distance of paired ray points / scene_radius',lambda_geometry=args.lambda_geometry,
            initial_model_fingerprint=model_fingerprint(g),selection=selection,
            pool_sha256=canonical_hash(groups),schedule_sha256=canonical_hash(schedule),
            prepare_seconds=time.monotonic()-started,selected_camera_count=len({i for q in groups for i in q['camera_indices']}),
            selection_settings={k:getattr(args,k) for k in ('neighbors','max_groups','candidate_limit','anchors_per_group','min_groups')},
            experiment_label='common fixed COLMAP anchors and joint availability constraints; not untouched native baseline')
        if sha256(identity['input_ply'])!=identity['input_sha256']:raise RuntimeError('input PLY changed during preparation')
        write_json(out/'experiment.json',manifest);write_json(out/'prepare_status.json',dict(status='complete',
            experiment_sha256=sha256(out/'experiment.json'),prepare_seconds=manifest['prepare_seconds'],
            head=head,field_protocol=field_protocol,geometry_protocol=protocol,native_config=config,
            resolution=metadata['resolution'],runtime_identity=manifest['runtime_identity']))
        print(json.dumps(dict(experiment=str(out/'experiment.json'),sha256=sha256(out/'experiment.json'))),flush=True)
    except Exception as exc:
        write_json(out/'prepare_status.json',dict(status='failed',error=repr(exc),seconds=time.monotonic()-started,
            diagnostic=getattr(exc,'diagnostic',None),actual_optimizer_steps=0,head=head,
            field_protocol=field_protocol,geometry_protocol=protocol,native_config=config,
            resolution=args.resolution,runtime_identity=manifest['runtime_identity']));raise


def optimizer_steps(g):
    values={}
    for group in g.optimizer.param_groups:
        steps=[int(g.optimizer.state.get(p,{}).get('step',0)) for p in group['params']]
        values[group['name']]=steps
    return values


def save_snapshot(g,out,completed,experiment,manifest_hash,arm):
    path=out/'point_cloud'/('iteration_'+str(7000+completed))
    path.mkdir(parents=True,exist_ok=False);g.save_ply(str(path/'point_cloud.ply'))
    write_json(path/'snapshot.json',dict(ply_sha256=sha256(path/'point_cloud.ply'),completed_updates=completed,
        global_iteration=7000+completed,arm=arm,input_identity=experiment['input_identity'],
        experiment_manifest_sha256=manifest_hash,geometry_protocol=experiment['geometry_protocol'],
        field_protocol=experiment['field_protocol'],native_config=experiment['native_config'],
        training_head=experiment['head'],resolution=experiment['camera_metadata']['resolution'],
        runtime_identity=experiment['runtime_identity'],
        optimizer_steps=optimizer_steps(g),fresh_adam=True,post_update_validated=True,
        appearance_initialization=experiment['appearance_initialization'],
        appearance_model=g.app_model.value,current_model_fingerprint=model_fingerprint(g)))
    # Model.capture omits filter_3D; include it explicitly for any future resume.
    torch.save(dict(model_capture=g.capture(),filter_3D=g.filter_3D.detach(),
                    completed_updates=completed,arm=arm,experiment_sha256=manifest_hash,
                    training_head=experiment['head'],field_protocol=experiment['field_protocol'],
                    geometry_protocol=experiment['geometry_protocol'],native_config=experiment['native_config'],
                    resolution=experiment['camera_metadata']['resolution'],runtime_identity=experiment['runtime_identity']),
               path/'checkpoint.pth')


def train(args):
    head=checked_source(args.expected_head);manifest_path=Path(args.experiment).resolve()
    if sha256(manifest_path)!=args.experiment_sha256:raise ValueError('experiment manifest hash mismatch')
    experiment=json.loads(manifest_path.read_text())
    if experiment['status']!='complete':raise ValueError('unfinished preparation')
    if experiment.get('schema')!='fixed_colmap_joint_expected_v1' or experiment['head']!=head:
        raise ValueError('preparation/source version mismatch')
    if experiment.get('field_protocol')!=FIELD_PROTOCOL:raise ValueError('shared field protocol changed since preparation')
    if experiment.get('resolution')!=experiment['camera_metadata']['resolution']:
        raise ValueError('declared/camera resolution mismatch')
    if experiment['runtime_identity']!=runtime_identity():raise ValueError('native CUDA runtime identity changed')
    if canonical_hash(experiment['groups'])!=experiment['pool_sha256'] or canonical_hash(experiment['schedule'])!=experiment['schedule_sha256']:
        raise ValueError('pool/schedule hash mismatch')
    steps=experiment['updates'] if args.steps is None else args.steps
    if not 1<=steps<=len(experiment['schedule']):raise ValueError('requested steps exceed frozen schedule')
    original=experiment['input_identity']
    identity=verify_inputs(original['input_ply'],original['source_path'],original['model_id'])
    config=native_config(experiment['native_config']['path'])
    if config!=experiment['native_config']:raise ValueError('native config changed')
    out=create_output(args.output);start=time.monotonic();seed_all(experiment['seed'])
    def deadline():
        if time.monotonic()-start>args.max_seconds:raise TimeoutError('training wall budget exhausted')
        if shutil.disk_usage(out).free<1<<30:raise RuntimeError('system output disk below1GiB free')
    report=dict(status='started_not_complete',head=head,arm=args.arm,input_identity=identity,
        experiment_manifest_sha256=args.experiment_sha256,experiment_path=str(manifest_path),
        pool_sha256=experiment['pool_sha256'],schedule_sha256=experiment['schedule_sha256'],
        target_updates=steps,completed_updates=0,optimizer_updates_executed=0,
        fresh_adam=True,full_training_state_restored=False,base_iteration=7000,
        pilot=steps<experiment['updates'],geometry_protocol=experiment['geometry_protocol'],
        field_protocol=experiment['field_protocol'],training_head=head,native_config=config,
        resolution=experiment['camera_metadata']['resolution'],runtime_identity=experiment['runtime_identity'])
    report['appearance_initialization']=experiment['appearance_initialization']
    write_json(out/'run_manifest.json',report);journal=out/'updates.jsonl';journal.touch(exist_ok=False)
    g=None;completed=0;attempted=0;previous=None;previous_optimizer=None;timings=[]
    try:
        cameras,metadata=load_cameras(original['source_path'],experiment['camera_metadata']['resolution'],'cpu',
                                     train_only=True,eval_split=config['settings']['eval'])
        if canonical_hash(metadata)!=canonical_hash(experiment['camera_metadata']):raise ValueError('camera/image metadata changed')
        g=load_model(identity,config['settings'],metadata['scene_radius'],len(cameras),True,experiment['optimizer_settings'])
        if model_fingerprint(g)!=experiment['initial_model_fingerprint']:raise ValueError('initial arrays differ from frozen experiment')
        filter_hash=hashlib.sha256(g.filter_3D.cpu().numpy().tobytes()).hexdigest();count=len(g.get_xyz)
        full_pool_check(g,cameras,experiment,deadline)
        save_snapshot(g,out,0,experiment,args.experiment_sha256,args.arm)
        checkpoints={x for x in (100,500,1000,steps) if x<=steps}
        torch.cuda.reset_peak_memory_stats()
        for update in range(1,steps+1):
            deadline();torch.cuda.synchronize();begin=time.perf_counter()
            entry=experiment['schedule'][update-1];group=experiment['groups'][entry['group_index']]
            ii=entry['anchor_indices'];g.update_learning_rate(7000+update)
            g.optimizer.zero_grad(set_to_none=True)
            ev=evaluate_pairs(g,cameras,group,ii,config['settings'],experiment['geometry_protocol'],
                joint_grad=args.arm=='joint',native_grad=True)
            require_available(ev,group,'pre_update')
            side=entry['rgb_side'];cam=cameras[group['camera_indices'][side]];pkg=ev['packages'][side]
            gt=cam.original_image.cuda();image=pkg['render'];weight=experiment['optimizer_settings']['lambda_dssim']
            rgb=(1-weight)*L1_loss_appearance(image,gt,g,cam.uid)+weight*(1-ssim(image[None],gt[None]))
            depth=ev[args.arm];x=ev['origins']+ev['directions']*depth[:,None];k=len(ii)
            geo=torch.linalg.vector_norm(x[:k]-x[k:],dim=-1).mean()/metadata['scene_radius']
            loss=rgb+experiment['lambda_geometry']*geo
            if not bool(torch.isfinite(loss)):raise GuardFailure('nonfinite loss',dict(update=update,rgb=rgb,geometry=geo))
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for q in g.optimizer.param_groups for p in q['params']):
                raise GuardFailure('nonfinite gradient',dict(update=update))
            # Preserve exact last valid parameters + Adam state before a possible
            # failed update. No continuing/backtracking after failure is allowed.
            previous={name:getattr(g,name).detach().clone() for name in model_parameter_names(g)}
            previous_optimizer=copy.deepcopy(g.optimizer.state_dict())
            g.optimizer.step();attempted+=1
            with torch.no_grad():
                if any(not bool(torch.isfinite(getattr(g,name)).all()) for name in model_parameter_names(g)):
                    raise GuardFailure('nonfinite updated parameter',dict(update=update))
                post=evaluate_pairs(g,cameras,group,ii,config['settings'],experiment['geometry_protocol'])
                require_available(post,group,'post_update')
                if update%experiment['full_pool_every']==0 or update==steps:
                    full_pool_check(g,cameras,experiment,deadline)
            completed=update;previous=None;previous_optimizer=None
            if len(g.get_xyz)!=count:raise AssertionError('topology changed')
            if optimizer_steps(g).get('xyz')!=[completed]:raise AssertionError('actual Adam step counter mismatch')
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin;timings.append(elapsed)
            row=dict(update=completed,global_iteration=7000+completed,optimizer_updates_executed=attempted,
                rgb=float(rgb.detach()),geometry=float(geo.detach()),loss=float(loss.detach()),seconds=elapsed,
                group_index=entry['group_index'],joint_valid=int(post['joint_valid'].sum()),
                native_valid=int(post['native_valid'].sum()),rays=2*k,optimizer_steps=optimizer_steps(g))
            with journal.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
            if completed in checkpoints:save_snapshot(g,out,completed,experiment,args.experiment_sha256,args.arm)
            del ev,post,loss,rgb,geo,depth,x,pkg,image,gt
        if hashlib.sha256(g.filter_3D.cpu().numpy().tobytes()).hexdigest()!=filter_hash:raise AssertionError('stored3D filter changed')
        if sha256(identity['input_ply'])!=identity['input_sha256']:raise RuntimeError('original PLY changed')
        report.update(status='complete',completed_updates=completed,optimizer_updates_executed=attempted,
            seconds=time.monotonic()-start,step_seconds=timings,optimizer_steps=optimizer_steps(g),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),original_ply_unchanged=True)
        write_json(out/'run_manifest.json',report)
    except Exception as exc:
        # If a post-step guard failed, restore the exact pre-step state solely
        # for archival. The failed attempted update stays counted in the report.
        if g is not None:
            if previous is not None:
                with torch.no_grad():
                    for name,value in previous.items():getattr(g,name).copy_(value)
                g.optimizer.load_state_dict(previous_optimizer)
            last=out/'last_accepted_before_failure';last.mkdir(exist_ok=False)
            g.save_ply(str(last/'point_cloud.ply'))
            torch.save(dict(model_capture=g.capture(),filter_3D=g.filter_3D.detach(),
                completed_updates=completed,experiment_sha256=args.experiment_sha256,training_head=head,
                field_protocol=experiment['field_protocol'],geometry_protocol=experiment['geometry_protocol'],
                native_config=config,resolution=metadata['resolution'],runtime_identity=experiment['runtime_identity']),
                last/'checkpoint.pth')
            write_json(last/'state.json',dict(completed_updates=completed,optimizer_updates_attempted=attempted,
                is_successful_final_result=False,last_accepted_state_only=True,
                training_head=head,field_protocol=experiment['field_protocol'],geometry_protocol=experiment['geometry_protocol'],
                native_config=config,resolution=metadata['resolution'],runtime_identity=experiment['runtime_identity'],
                caveat='not all fixed-pool rays were necessarily available at failure'))
        report.update(status='failed',completed_updates=completed,optimizer_updates_executed=attempted,
            error=repr(exc),diagnostic=getattr(exc,'diagnostic',None),seconds=time.monotonic()-start)
        write_json(out/'run_manifest.json',report);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--expected-head',required=True);common.add_argument('--output',required=True)
    common.add_argument('--max-seconds',type=float,required=True)
    prep=sub.add_parser('prepare',parents=[common]);prep.add_argument('--input-ply',required=True)
    prep.add_argument('--source-path',required=True);prep.add_argument('--native-config',required=True)
    prep.add_argument('--model-id',choices=sorted(MODEL_IDENTITIES),required=True)
    prep.add_argument('--updates',type=int,default=1000);prep.add_argument('--seed',type=int,default=713)
    prep.add_argument('--resolution',type=int,default=2);prep.add_argument('--pairs-per-step',type=int,default=8)
    prep.add_argument('--anchors-per-group',type=int,default=16);prep.add_argument('--max-groups',type=int,default=32)
    prep.add_argument('--min-groups',type=int,default=8);prep.add_argument('--neighbors',type=int,default=2)
    prep.add_argument('--candidate-limit',type=int,default=128);prep.add_argument('--lambda-geometry',type=float,default=.02)
    prep.add_argument('--member-epsilon',type=float,default=1e-6);prep.add_argument('--max-members',type=int,default=1024)
    prep.add_argument('--min-native-alpha',type=float,default=.8);prep.add_argument('--max-initial-pair-gap',type=float,default=.05)
    prep.add_argument('--full-pool-every',type=int,default=100)
    tr=sub.add_parser('train',parents=[common]);tr.add_argument('--experiment',required=True)
    tr.add_argument('--experiment-sha256',required=True);tr.add_argument('--arm',choices=('expected','joint'),required=True)
    tr.add_argument('--steps',type=int)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if not 0<args.max_seconds<=21600:raise ValueError('explicit wall budget must be <=6h')
    if args.command=='prepare':
        if not 1<=args.updates<=23000 or not 1<=args.pairs_per_step<=16:raise ValueError('invalid update/ray budget')
        if not args.pairs_per_step<=args.anchors_per_group<=args.candidate_limit<=512:raise ValueError('invalid pool sizes')
        if not 1<=args.min_groups<=args.max_groups<=128:raise ValueError('invalid group counts')
        if not 0<args.member_epsilon<1 or not 1<=args.max_members<=2048:raise ValueError('invalid support budget')
        if not 0<=args.min_native_alpha<=1 or not 0<args.max_initial_pair_gap<=1:raise ValueError('invalid anchor gate')
        if not 0<=args.lambda_geometry<=1:raise ValueError('invalid geometry weight')
        if args.full_pool_every<1:raise ValueError('positive pool-check interval required')
        prepare(args)
    else:
        train(args)

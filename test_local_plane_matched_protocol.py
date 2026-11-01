"""CPU/mock gates only; never represent a host CUDA training run."""
import ast
import io
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

import numpy as np
import torch

from matched_plane_regularizer import FrozenPlaneRegularizer, EXPECTED_R0


ROOT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).with_name("train_local_plane_matched.py")


def load_pure(name,globals_dict):
    tree=ast.parse(SCRIPT.read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=compile(ast.Module(body=[node],type_ignores=[]),str(SCRIPT),"exec")
    scope=dict(globals_dict)
    exec(code,scope)
    return scope[name]


class Camera:
    def __init__(self,name,neighbors):self.image_name=name;self.nearest_id=neighbors


class FakeTorch:
    @staticmethod
    def tensor(value,**kwargs):
        kwargs.pop("device",None)
        return torch.tensor(value,**kwargs)
    isfinite=staticmethod(torch.isfinite)
    no_grad=staticmethod(torch.no_grad)


class MockGaussians:
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.tensor(1.))
        self._xyz=self.parameter
        self._rotation=self.parameter
        self.optimizer=torch.optim.Adam([{"params":[self.parameter],"lr":.001,"name":"xyz"}])
        self.virtual=[];self.sh_calls=0
    def update_learning_rate(self,i):self.virtual.append(i)
    def oneupSHdegree(self):self.sh_calls+=1


class Controls(unittest.TestCase):
    def test_initial_parameter_and_adam_identity(self):
        signature=load_pure("snapshot_hash",{"hashlib":__import__("hashlib")})
        optimizer_hash=load_pure("initial_optimizer_hash",{"hashlib":__import__("hashlib"),"json":json})
        class State:
            active_sh_degree=3
            def __init__(self):
                for name in ("_xyz","_features_dc","_features_rest","_scaling",
                             "_rotation","_opacity","filter_3D","_appearance_embeddings"):
                    setattr(self,name,torch.zeros((2,3)))
                self.optimizer=torch.optim.Adam([{"params":[self._xyz],"lr":.001,"name":"xyz"}])
        a,b=State(),State()
        self.assertEqual(signature(a),signature(b))
        self.assertEqual(optimizer_hash(a),optimizer_hash(b))
        b._rotation[0,0]=1
        self.assertNotEqual(signature(a),signature(b))

    def test_lambda_and_fixed_schedule(self):
        lam=load_pure("lambda_from_initial",{"np":np,"ValueError":ValueError})
        self.assertAlmostEqual(lam(.2,140.34713758283957),.002/140.34713758283957)
        for rgb,R in ((float("nan"),1.),(1.,0.)):
            with self.assertRaises(ValueError):lam(rgb,R)
        schedule=load_pure("fixed_schedule",{"random":random,"UPDATES":1000})
        cameras=[Camera("a",[1]),Camera("b",[0])]
        a=schedule(cameras);b=schedule(cameras)
        self.assertEqual(a,b)
        self.assertEqual(len(a),1000)
        self.assertEqual({x[0] for x in a},{0,1})
        self.assertTrue(all(y==1-x for x,y in a))

    def test_exact_1000_updates_and_boundaries(self):
        saved=[]
        def boundary(scene,gaussians,arm_dir,update,filter_hash):
            saved.append((update,gaussians.virtual[-1] if gaussians.virtual else None,
                          float(gaussians.parameter.detach())))
            return dict(completed_updates=update,iteration=7000+update)
        def host(*args):
            p=args[0].parameter
            value=p.square()
            return value,dict(rgb=value,normal=value,ncc=value,geo=value)
        class MockPath:
            def __init__(self,value):self.value=value
            def __truediv__(self,other):return self
            def open(self,*args,**kwargs):return io.StringIO()
        train=load_pure("train_arm",dict(torch=FakeTorch,PatchMatch=lambda *a,**k:object(),
            save_boundary=boundary,host_step_loss=host,UPDATES=1000,
            time=time,Path=MockPath,json=json))
        cameras=[Camera("a",[1]),Camera("b",[0])]
        schedule=[(0,1)]*1000
        class Dataset:white_background=False;kernel_size=0.
        class Opt:multi_view_patch_size=3;multi_view_pixel_noise_th=1.
        g=MockGaussians()
        reports=train("B",object(),g,cameras,schedule,lambda *x:g.parameter.square(),
            .01,object(),Dataset,Opt,"unused",time.monotonic()+60,"frozen")
        self.assertEqual(g.virtual,list(range(7001,8001)))
        self.assertEqual(g.sh_calls,1)
        self.assertEqual([r["completed_updates"] for r in reports],[0,500,1000])
        self.assertEqual([x[0] for x in saved],[0,500,1000])
        self.assertEqual([x[1] for x in saved],[None,7500,8000])
        self.assertLess(saved[-1][2],saved[-2][2])
        self.assertEqual(int(g.optimizer.state[g.parameter]["step"]),1000)
        a=MockGaussians()
        def forbidden_regularizer(*args):
            raise AssertionError("A must not call the new regularizer")
        train("A",object(),a,cameras,schedule,forbidden_regularizer,.01,
              object(),Dataset,Opt,"unused",time.monotonic()+60,"frozen")
        self.assertEqual(int(a.optimizer.state[a.parameter]["step"]),1000)

    def test_expected_host_formula(self):
        def rgb(*args):return torch.tensor(.7)
        def normal(*args):return torch.zeros((3,2,2))
        def render(*args,**kwargs):
            return dict(render=torch.zeros((3,2,2)),expected_depth=torch.ones((1,2,2)),
                        median_depth=torch.full((1,2,2),123.),normal=torch.ones((3,2,2)))
        class Patch:
            def __call__(self,gaussians,pkg,cam,neighbor,it,depth_normal):
                assert torch.equal(pkg["expected_depth"],torch.ones((1,2,2)))
                assert torch.equal(depth_normal,torch.zeros((3,2,2)))
                return torch.tensor(.4),torch.tensor(.2)
        host=load_pure("host_step_loss",dict(torch=torch,render=render,
            host_rgb_loss=rgb,depth_to_normal=normal))
        class Opt:lambda_depth_normal=.05;lambda_multi_view_ncc=.3;lambda_multi_view_geo=.02
        loss,parts=host(None,None,None,None,None,0.,Patch(),Opt,7001)
        self.assertAlmostEqual(float(loss),.7+.05*1+.3*.4+.02*.2,places=6)
        self.assertAlmostEqual(float(parts["rgb"]),.7,places=6)

    def test_real_capture_R_and_shared_gradient(self):
        cap=ROOT/"G5_scan110_native_capture_faa9294/native_members.npz"
        result=ROOT/"G5_local_plane_training_cpu_v1/results_v4.json"
        camera=ROOT/"G5_scan110_native_capture_faa9294_bundle/camera"
        if not (cap.exists() and result.exists() and camera.exists()):
            self.skipTest("fixed local capture unavailable")
        reg=FrozenPlaneRegularizer(cap,result,camera)
        with np.load(cap,allow_pickle=False) as z:
            xyz=torch.zeros((142892,3),requires_grad=True)
            quat=torch.zeros((142892,4),requires_grad=True)
            with torch.no_grad():
                quat[:,0]=1
                xyz[z["unique_id"]]=torch.from_numpy(z["center"])
                quat[z["unique_id"]]=torch.from_numpy(z["raw_quaternion"])
        R=reg(xyz,quat)
        self.assertAlmostEqual(float(R),EXPECTED_R0,places=9)
        R.backward()
        self.assertTrue(bool(torch.isfinite(xyz.grad).all()))
        self.assertTrue(bool(torch.isfinite(quat.grad).all()))
        self.assertGreater(int(torch.count_nonzero(xyz.grad)),0)
        self.assertEqual(int(torch.count_nonzero(xyz.grad[~torch.isin(torch.arange(142892),reg.ids)])),0)


if __name__=="__main__":unittest.main()

"""CPU/mock gates only; never represent a host CUDA training run."""
import ast
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import types
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
    isposinf=staticmethod(torch.isposinf)
    no_grad=staticmethod(torch.no_grad)


class MockGaussians:
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.tensor(1.))
        self._xyz=self.parameter
        self._rotation=self.parameter
        self._opacity=torch.nn.Parameter(torch.zeros((1,1)))
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
            time=time,Path=MockPath,json=json,
            verify_updated_parameters=lambda *args:None))
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

    def test_cuda_device_resolution(self):
        self.assertEqual(torch.device("cpu"),torch.empty(1,device="cpu").device)
        if torch.cuda.is_available():
            requested=torch.device("cuda")
            actual=torch.empty(1,device=requested).device
            self.assertEqual(actual,torch.device("cuda",torch.cuda.current_device()))
            self.assertNotEqual(requested,actual)

    def test_frozen_configuration_rejects_cli_drift(self):
        tree=ast.parse(SCRIPT.read_text(encoding="utf-8"))
        names=("FROZEN_OPT","FROZEN_DATASET","FROZEN_PIPE")
        assignments=[n for n in tree.body if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
        scope={}
        exec(compile(ast.Module(body=assignments,type_ignores=[]),str(SCRIPT),"exec"),scope)
        verify=load_pure("verify_frozen_configuration",scope)
        class Settings:
            def __init__(self,items):self.__dict__.update(items)
        opt=Settings(scope["FROZEN_OPT"]);data=Settings(scope["FROZEN_DATASET"])
        data.source_path="source";data.model_path="output"
        pipe=Settings(scope["FROZEN_PIPE"])
        self.assertEqual(verify(data,opt,pipe)["optimization"]["rotation_lr"],.001)
        opt.rotation_lr=.002
        with self.assertRaises(ValueError):verify(data,opt,pipe)
        opt.rotation_lr=.001;data.multi_view_num=4
        with self.assertRaises(ValueError):verify(data,opt,pipe)

    def test_positive_inf_opacity_boundary_and_adam(self):
        activated=load_pure("verify_activated_parameters",{"torch":torch})
        initial=load_pure("initial_opacity_boundary",{"torch":torch,
            "verify_activated_parameters":activated})
        updated=load_pure("verify_updated_parameters",{"torch":torch,
            "verify_activated_parameters":activated})
        class G:
            _opacity=torch.nn.Parameter(torch.tensor([[float("inf")],[0.]]))
            _xyz=torch.nn.Parameter(torch.ones((2,3)))
            _scaling=torch.nn.Parameter(torch.zeros((2,3)))
            _rotation=torch.nn.Parameter(torch.ones((2,4)))
            _features_dc=torch.zeros(2,1)
            _features_rest=torch.zeros(2,1)
            optimizer=torch.optim.Adam([{"params":[_opacity],"lr":.05,"name":"opacity"}])
            @property
            def get_opacity(self):return torch.sigmoid(self._opacity)
            @property
            def get_opacity_with_3D_filter(self):
                square=self.get_scaling.square()
                coef=torch.sqrt(square.prod(dim=1)/(square+.1).prod(dim=1))
                return self.get_opacity*coef[:,None]
            @property
            def get_scaling(self):return torch.exp(self._scaling)
            @property
            def get_scaling_with_3D_filter(self):return torch.sqrt(self.get_scaling.square()+.1)
        g=G();allowed=initial(g)
        self.assertEqual(float(g.get_opacity[0,0]),1.)
        g.get_opacity.sum().backward()
        self.assertTrue(bool(torch.isfinite(g._opacity.grad).all()))
        self.assertEqual(float(g._opacity.grad[0,0]),0.)
        g.optimizer.step();updated(g,allowed,7001)
        self.assertTrue(bool(torch.isposinf(g._opacity[0,0])))
        with torch.no_grad():g._opacity[1,0]=float("inf")
        with self.assertRaises(ValueError):updated(g,allowed,7002)

    def test_extension_source_inventory_covers_sample_wrapper_and_glm(self):
        tree=ast.parse(SCRIPT.read_text(encoding="utf-8"))
        names=("OWN_EXTENSION_SOURCE_DIRS","EXTENSION_SOURCE_SUFFIXES")
        assignments=[n for n in tree.body if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
        scope={"Path":Path,"__file__":str(SCRIPT),"sha256":lambda p:hashlib.sha256(p.read_bytes()).hexdigest()}
        exec(compile(ast.Module(body=assignments,type_ignores=[]),str(SCRIPT),"exec"),scope)
        inventory=load_pure("extension_sources",scope)
        if not any((SCRIPT.parent/"submodules/diff-gaussian-rasterization/third_party/glm").glob("**/*.hpp")):
            with self.assertRaisesRegex(ValueError,"extension_sample_wrapper_or_glm_source_missing"):
                inventory()
            return  # Local submodule is absent; deployment must populate and verify it.
        sources=inventory()
        self.assertTrue(any(p.endswith("sample_forward.cu") for p in sources))
        self.assertTrue(any(p.endswith("ext.cpp") for p in sources))
        self.assertTrue(any("third_party/glm/" in p for p in sources))

    def test_wrapper_binding_accepts_top_level_backend_and_ignores_unrelated(self):
        backend=types.ModuleType("fused_ssim_cuda")
        backend.__file__="C:/fake/fused_ssim_cuda.pyd"
        unrelated=types.ModuleType("unrelated_cuda")
        unrelated.__file__="C:/fake/unrelated_cuda.pyd"
        scope={"__name__":"fused_ssim","fused_ssim_cuda":backend}
        exec("class FusedAutograd:\n"
             "    @staticmethod\n"
             "    def forward(): return fused_ssim_cuda.forward()\n"
             "    @staticmethod\n"
             "    def backward(): return fused_ssim_cuda.backward()\n"
             "def wrapper(): return FusedAutograd.forward()\n",scope)
        resolver=load_pure("native_modules_referenced_by_wrapper",dict(
            inspect=__import__("inspect"),sys=sys,
            importlib=__import__("importlib")))
        before={k:sys.modules.get(k) for k in ("fused_ssim_cuda","unrelated_cuda")}
        try:
            sys.modules["fused_ssim_cuda"]=backend
            sys.modules["unrelated_cuda"]=unrelated
            self.assertEqual(list(resolver(scope["wrapper"])),["fused_ssim_cuda"])
            del scope["fused_ssim_cuda"]
            self.assertEqual(resolver(scope["wrapper"]),{})
        finally:
            for key,value in before.items():
                if value is None:sys.modules.pop(key,None)
                else:sys.modules[key]=value

    def test_binary_manifest_mismatch_is_rejected(self):
        # The same package name cannot pass with a different compiled hash.
        class ManifestPath:
            def read_text(self,encoding=None):return "ignored"
        verify=load_pure("verify_build_identity",dict(__file__=str(SCRIPT),
            Path=lambda value:ManifestPath() if isinstance(value,ManifestPath) else Path(value),
            subprocess=types.SimpleNamespace(check_output=lambda *a,**k:"reviewed\n"),
            json=types.SimpleNamespace(loads=lambda text:{"reviewed_head":"reviewed",
                "bindings":{"fused_ssim":{"backends":[{"sha256":"wrong"}]}}}),
            loaded_extension_bindings=lambda:{"fused_ssim":{"backends":[{"sha256":"actual"}]}}))
        with self.assertRaisesRegex(ValueError,"extension_binary_identity_mismatch"):
            verify("reviewed",ManifestPath())

    def test_unverified_historical_packages_still_require_binding_and_environment(self):
        bindings={name:{"wrapper":name,"backends":[{"module":name+"_cuda",
            "path":"C:/fake/"+name+".pyd","sha256":"binary"}]}
            for name in ("rasterizer","warp","simple_knn","fused_ssim")}
        manifest=dict(reviewed_head="reviewed",bindings=bindings,sources={"file.cu":"source"},
            environment=dict(python=sys.version,torch=torch.__version__,
                             torch_cuda=str(torch.version.cuda)),
            own_build_provenance={name:dict(source_head="reviewed",build_command="build",
                build_evidence_path="evidence",build_evidence_sha256="evidence-sha")
                for name in ("rasterizer","warp")},
            historical_provenance=dict(fused_ssim="unverified",simple_knn="unverified"))
        manifest["own_build_provenance"]["rasterizer"]["glm_head"]="reviewed"
        class P:
            def __init__(self,value):self.value=str(value)
            @property
            def parent(self):return self
            def __truediv__(self,other):return self
            def read_text(self,encoding=None):return "manifest"
            def resolve(self):return self
            def is_file(self):return True
        def check_output(*args,**kwargs):return "reviewed\n"
        verify=load_pure("verify_build_identity",dict(__file__=str(SCRIPT),Path=P,
            subprocess=types.SimpleNamespace(check_output=check_output),
            json=types.SimpleNamespace(loads=lambda text:manifest),
            loaded_extension_bindings=lambda:bindings,extension_sources=lambda:{"file.cu":"source"},
            OWN_EXTENSION_SOURCE_DIRS=("raster","warp"),sha256=lambda path:"evidence-sha",
            sys=sys,torch=torch))
        self.assertEqual(verify("reviewed","manifest")["historical_provenance"],
                         manifest["historical_provenance"])
        manifest["environment"]["torch"]="wrong"
        with self.assertRaisesRegex(ValueError,"extension_environment_identity_mismatch"):
            verify("reviewed","manifest")

    def test_boundary_failed_before_save_is_partial(self):
        files={};directories=set()
        class P:
            def __init__(self,value):self.value=str(value)
            def __truediv__(self,key):return P(self.value+"/"+str(key))
            def __str__(self):return self.value
            @property
            def parent(self):return P(self.value.rsplit("/",1)[0])
            def mkdir(self,parents=False,exist_ok=False):directories.add(self.value)
            def is_file(self):return self.value in files
            def read_text(self,encoding=None):return files[self.value]
        class SavingTorch:
            @staticmethod
            def save(value,path):files[str(path)]=value
        save=load_pure("save_failure_state",{"Path":P,"json":json,"torch":SavingTorch,
            "SAVED_UPDATES":(0,500,1000)})
        class G:
            filter_3D=torch.ones(1)
            def capture(self):return {"state":1}
            def save_ply(self,path):files[path]=b"partial"
        result=save(None,G(),"arm",500)
        self.assertEqual(result,"arm/partial_7500.pth")
        self.assertIn("arm/partial/iteration_7500/point_cloud.ply",files)
        self.assertTrue(files[result]["partial"])
        files["arm/complete_8000.json"]=json.dumps(dict(completed_updates=1000,
            ply="arm/point_cloud/iteration_8000/point_cloud.ply",checkpoint="arm/chkpnt8000.pth"))
        # A stale marker cannot turn a failed 1000th update into a complete result.
        self.assertEqual(save(None,G(),"arm",1000),"arm/partial_8000.pth")


if __name__=="__main__":unittest.main()

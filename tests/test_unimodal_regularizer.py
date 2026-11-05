"""Segmented loss tests. Run python -m unittest discover -s tests -v.
CPU tests do not certify the CUDA renderer. Set GAUDEPTH_REQUIRE_CUDA_TESTS=1
for a required GPU run; missing CUDA or a local compiled extension then fails.
Legacy per-Gaussian suite is preserved in Git commit 344a16a.
"""
import importlib.machinery
import importlib.util
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('GAUDEPTH_REQUIRE_CUDA_TESTS') == '1' and (torch is None or not torch.cuda.is_available()):
    raise RuntimeError('Required CUDA PyTorch is unavailable')


def sign(x):
    return float(x > 0) - float(x < 0)


def mass_loss(mass, epsilon=1e-6, min_mass=0):
    total = sum(mass)
    if not mass or total <= 0 or total < min_mass:
        return 0., [0.] * len(mass)
    padded = [0.] + list(mass) + [0.]
    v = sum(abs(b-a) for a, b in zip(padded, padded[1:]))
    peak = max(mass)
    excess = max(0., v-2*peak)
    denom = 2*peak + epsilon*total
    if not excess:
        return 0., [0.] * len(mass)
    argmax = mass.index(peak)
    grad = []
    for k, x in enumerate(mass):
        dv = sign(x-padded[k]) + sign(x-padded[k+2])
        h = float(k == argmax)
        grad.append((dv-2*h)/denom - excess*(2*h+epsilon)/denom**2)
    return excess/denom, grad


def boundaries(raw, depths, bins=4, near=0., far=4.):
    result = [1.] * (bins+1)
    accepted = []
    original_t = 1.
    for i, (a, d) in enumerate(zip(raw, depths)):
        alpha = min(.99, a)
        if alpha < 1/255:
            continue
        next_t = original_t * (1-alpha)
        if next_t < 1e-4:
            break
        accepted.append((i, alpha, d))
        original_t = next_t
        if not math.isfinite(d):
            continue
        for j in range(bins+1):
            if d < near+(far-near)*j/bins:
                result[j] *= 1-alpha
    return result, accepted


def alpha_loss(raw, depths, bins=4, near=0., far=4., epsilon=1e-6, min_mass=0):
    t, accepted = boundaries(raw, depths, bins, near, far)
    mass = [a-b for a, b in zip(t, t[1:])]
    loss, gm = mass_loss(mass, epsilon, min_mass)
    gt = [0.] * len(t)
    for k, g in enumerate(gm):
        gt[k] += g
        gt[k+1] -= g
    ga = [0.] * len(raw)
    for i, a, d in accepted:
        if raw[i] >= .99 or not math.isfinite(d):
            continue
        ga[i] = -sum(g * tj for j, (g, tj) in enumerate(zip(gt, t))
                     if d < near+(far-near)*j/bins)/(1-a)
    return loss, ga, mass


def finite_diff(fn, x, step=1e-6):
    out=[]
    for k in range(len(x)):
        a, b = list(x), list(x)
        a[k]-=step; b[k]+=step
        out.append((fn(b)-fn(a))/(2*step))
    return out


class SegmentedMathTests(unittest.TestCase):
    def close(self, a, b, tol=2e-6):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, delta=tol)

    def test_single_peak_plateau_and_one_bin(self):
        for m in ([.1,.3,.2], [.1,.2,.2,.1], [.8], [.4,.3,.1], [0,.2,0]):
            self.assertAlmostEqual(mass_loss(m)[0], 0, delta=1e-14)

    def test_multiple_peaks_penalized(self):
        self.assertGreater(mass_loss([.4,.1,.3])[0], mass_loss([.4,.2,.3])[0])

    def test_normalization_matches_probability_formula(self):
        m=[.21,.03,.29,.08]
        p=[x/sum(m) for x in m]
        # A deliberately non-tiny epsilon catches omission of epsilon*A.
        self.assertAlmostEqual(mass_loss(m, .2)[0], mass_loss(p, .2)[0], places=13)

    def test_mass_gradient(self):
        m=[.21,.03,.29,.08]
        self.close(mass_loss(m,.2)[1], finite_diff(lambda x:mass_loss(x,.2)[0],m))

    def test_alpha_gradient_all_boundaries(self):
        raw=[.11,.25,.07,.41,.13,.18]
        depths=[.2,.6,1.4,2.5,3.2,3.7]
        value,g,_=alpha_loss(raw,depths,epsilon=.2)
        self.assertGreater(value,0)
        self.close(g,finite_diff(lambda x:alpha_loss(x,depths,epsilon=.2)[0],raw))

    def test_non_center_order_and_clipped_alpha(self):
        raw=[.2,.3,.11,1.2,.04]
        d=[2.5,.5,3.5,1.5,3.8]
        _,g,_=alpha_loss(raw,d)
        self.close(g,finite_diff(lambda a:alpha_loss(a,d)[0],raw),1e-5)
        self.assertEqual(g[3],0)

    def test_random_fixed_membership_gradients(self):
        rng=random.Random(981)
        for _ in range(80):
            a=[rng.uniform(.05,.35) for _ in range(7)]
            d=[rng.randrange(4)+.5 for _ in a]
            self.close(alpha_loss(a,d)[1],finite_diff(lambda x:alpha_loss(x,d)[0],a),1e-5)

    def test_split_inside_bin_preserves_mass(self):
        # Two factors whose product equals the original factor.
        a=.4; split=1-math.sqrt(1-a)
        m1=alpha_loss([a,.2,.3],[.3,1.5,2.5])[2]
        m2=alpha_loss([split,split,.2,.3],[.2,.4,1.5,2.5])[2]
        self.close(m1,m2,1e-14)

    def test_zero_mass_and_empty_bins(self):
        self.assertEqual(mass_loss([0,0,0])[0],0)
        self.assertEqual(alpha_loss([.5],[7.])[0],0)
        self.assertGreater(mass_loss([.4,0,.3])[0],0)

    def test_edge_convention_and_range(self):
        t,_=boundaries([.2,.3,.4],[-.5,1.,4.])
        self.close(t,[.8,.8,.56,.56,.56],1e-14)

    def test_tie_subgradient(self):
        g=mass_loss([.3,.1,.3])[1]
        self.assertLess(g[0],g[2])

    @unittest.skipUnless(shutil.which('g++'),'Host C++ compiler unavailable')
    def test_actual_cuda_arithmetic_header_on_host(self):
        # Compile the actual shared header as host C++; this checks arithmetic,
        # not CUDA kernel compilation or device resource limits.
        header=ROOT/'submodules/diff-gaussian-rasterization/cuda_rasterizer/segmented_unimodal.cuh'
        source='#define __device__\n#include "'+str(header)+'"\n#include <cstdio>\nint main(){float t[]={1,.79,.76,.47,.39};float v,m;int k;segmentStats(t,4,v,m,k);float g[5];segmentBoundaryGradients(t,4,.2f,1.f,g);printf("%.9g %.9g %d",v,m,k);for(float x:g)printf(" %.9g",x);}'
        with tempfile.TemporaryDirectory() as td:
            cpp=Path(td)/'test.cpp'; exe=Path(td)/'test'
            cpp.write_text(source)
            subprocess.run(['g++','-std=c++17',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
            vals=subprocess.check_output([str(exe)],text=True).split()
        t=[1,.79,.76,.47,.39]
        mass=[a-b for a,b in zip(t,t[1:])]
        gm=mass_loss(mass,.2)[1]; gt=[0.]*5
        for j,g in enumerate(gm):gt[j]+=g;gt[j+1]-=g
        self.assertEqual(int(vals[2]),2)
        self.close([float(x) for x in vals[3:]],gt,2e-6)


@unittest.skipIf(torch is None or not torch.cuda.is_available(),'CUDA PyTorch unavailable; device validation NOT performed')
class CudaSegmentedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=ROOT/'submodules/diff-gaussian-rasterization'
        package=root/'diff_gaussian_rasterization'
        override=os.environ.get('GAUDEPTH_RASTERIZER_BUILD')
        dirs=[Path(override)] if override else [package]+list((root/'build').glob('lib*/diff_gaussian_rasterization'))
        found=[d for d in dirs if any((d/('_C'+s)).is_file() for s in importlib.machinery.EXTENSION_SUFFIXES)]
        if not found:
            if os.environ.get('GAUDEPTH_REQUIRE_CUDA_TESTS')=='1':raise RuntimeError('Build current local CUDA extension first')
            raise unittest.SkipTest('No compiled local CUDA extension')
        if len(found)>1 and package not in found:raise RuntimeError('Select GAUDEPTH_RASTERIZER_BUILD')
        artifact=package if package in found else found[0]
        spec=importlib.util.spec_from_file_location('diff_gaussian_rasterization',package/'__init__.py',submodule_search_locations=[str(package),str(artifact)])
        cls.mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.mod;spec.loader.exec_module(cls.mod)

    def render(self,enabled,opacity,means=None,scales=None):
        tangent=math.tan(math.pi/6)
        projection=torch.zeros((4,4),device='cuda')
        projection[0,0]=projection[1,1]=1/tangent
        projection[3,2]=1
        projection[2,2]=100/(100-.01);projection[2,3]=-1/(100-.01)
        settings=self.mod.GaussianRasterizationSettings(
            image_height=64,image_width=64,tanfovx=tangent,tanfovy=tangent,kernel_size=0.,
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),
            projmatrix=projection.T.contiguous(),sh_degree=0,campos=torch.zeros(3,device='cuda'),
            prefiltered=False,require_depth=True,debug=True,unimodal_enabled=enabled,
            unimodal_min_mass=.5,unimodal_bins=4,unimodal_near=0.,unimodal_far=4.)
        if means is None:means=torch.tensor([[.01,.015,.6],[-.01,.01,1.6],[.015,-.01,2.6],[-.01,-.015,3.6]],device='cuda')
        if scales is None:scales=torch.tensor([[.12,.10,.05]]*4,device='cuda')
        return self.mod.GaussianRasterizer(settings)(means3D=means,means2D=torch.zeros_like(means),opacities=opacity,
            colors_precomp=torch.full((4,3),.3,device='cuda'),scales=scales,
            rotations=torch.tensor([[1.,0.,0.,0.]]*4,device='cuda'))

    def test_legacy_outputs_and_opacity_gradient(self):
        a=torch.tensor([[.26],[.08/.74],[.32/.66],[.04/.34]],device='cuda',requires_grad=True)
        off=self.render(False,a);on=self.render(True,a)
        self.assertEqual(len(off),6);self.assertEqual(len(on),8)
        for x,y in zip(off,on[:6]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        objective=lambda out:out[6][:,30:34,30:34].sum()
        self.assertTrue(bool(on[7][:,30:34,30:34].all()))
        self.assertGreater(objective(on).item(),0)
        objective(on).backward()
        num=finite_diff(lambda v:objective(self.render(True,torch.tensor(v,device='cuda').reshape(4,1))).item(),a.detach().flatten().tolist(),2e-4)
        for actual,expected in zip(a.grad.flatten().tolist(),num):self.assertAlmostEqual(actual,expected,delta=4e-3)

    def test_fixed_bin_geometry_gradient(self):
        a=torch.tensor([[.26],[.08/.74],[.32/.66],[.04/.34]],device='cuda')
        m=torch.tensor([[.01,.015,.6],[-.01,.01,1.6],[.015,-.01,2.6],[-.01,-.015,3.6]],device='cuda',requires_grad=True)
        s=torch.tensor([[.12,.10,.05]]*4,device='cuda',requires_grad=True)
        objective=lambda out:out[6][:,30:34,30:34].sum()
        out=self.render(True,a,m,s);objective(out).backward()
        for p,is_mean in ((m,True),(s,False)):
            vals=p.detach().flatten().tolist()
            def evaluate(v):
                changed=torch.tensor(v,device='cuda').reshape(4,3)
                return objective(self.render(True,a,changed if is_mean else m.detach(),s.detach() if is_mean else changed)).item()
            numerical=finite_diff(evaluate,vals,5e-4)
            self.assertTrue(bool(torch.isfinite(p.grad).all()))
            for x,y in zip(p.grad.flatten().tolist(),numerical):self.assertAlmostEqual(x,y,delta=4e-3)


if __name__=='__main__':unittest.main()

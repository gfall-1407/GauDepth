"""Host checks for the actual shared segmented CUDA arithmetic header.
These do not compile or execute a GPU kernel.
"""
import ctypes
import math
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/segmented_unimodal.cuh'

@unittest.skipUnless(shutil.which('g++'), 'host C++ compiler unavailable')
class StableUnimodalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        src = Path(cls.tmp.name) / 'wrapper.cpp'
        so = Path(cls.tmp.name) / 'wrapper.so'
        src.write_text('#define __device__\n#include "' + str(HEADER) + '"\n' + r'''
extern "C" void evaluate(const float* t,int n,float eps,float* out,float* grad) {
 float v,m; int p; segmentStats(t,n,v,m,p);
 float e=segmentExcess(t,n,p); float a=t[0]-t[n];
 out[0]=e;out[1]=m;out[2]=p;out[3]=(m>0 && a>0) ? e/(2*m+eps*a):0;out[4]=fmaxf(v-2*m,0.f);
 segmentBoundaryGradients(t,n,eps,1.f,grad);
}
''')
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',str(src),'-o',str(so)],check=True,capture_output=True,text=True)
        cls.lib=ctypes.CDLL(str(so));cls.fn=cls.lib.evaluate
        cls.fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_float,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def run_header(self,t,eps=.01):
        n=len(t)-1; T=(ctypes.c_float*(n+1))(*t); out=(ctypes.c_float*5)();g=(ctypes.c_float*(n+1))()
        self.fn(T,n,eps,out,g)
        return list(T),list(out),list(g)

    def test_unimodal_and_plateau_zero(self):
        for t in ([1,.875,.625,.125,0],[1,.75,.5,.25,0],[1,.5,0]):
            with self.subTest(t=t):
                _,out,g=self.run_header(t)
                self.assertEqual(out[0],0.0);self.assertEqual(out[3],0.0);self.assertTrue(all(x==0 for x in g))

    def test_two_peaks_and_exact_plateau_tie(self):
        _,out,g=self.run_header([1,.625,.5,.125,0])
        self.assertAlmostEqual(out[0],.5,places=7)
        self.assertTrue(all(math.isfinite(x) for x in g))
        # All-flat histogram has zero violation and chosen zero subgradient.
        _,out,g=self.run_header([1,.75,.5,.25,0]);self.assertEqual(out[3],0);self.assertEqual(g,[0]*5)

    def test_near_unimodal_true_violation_is_not_thresholded(self):
        _,out,_=self.run_header([1,.5,.25,.125+2**-16,0])
        self.assertGreater(out[0],0);self.assertLess(out[0],1e-3)

    def test_tied_peaks_follow_documented_hinge_subgradient(self):
        # masses [.25,.25,0,.25]: first maximum chosen; the flat edge
        # contributes zero hinge subgradient, while the last rise contributes.
        _,out,g=self.run_header([1.,.75,.5,.5,.25],.01)
        self.assertEqual(int(out[2]),0)
        den=.5+.01*.75; ex=.5
        dm=[-(ex*(2+.01))/(den*den), -ex*.01/(den*den),
            -2/den-ex*.01/(den*den), 2/den-ex*.01/(den*den)]
        expected=[dm[0],dm[1]-dm[0],dm[2]-dm[1],dm[3]-dm[2],-dm[3]]
        for got,want in zip(g,expected): self.assertAlmostEqual(got,want,delta=3e-6)

    def test_reproduced_float32_cancellation_cases(self):
        # These are represented float32 transmittances, not idealized masses.
        # A truly unimodal sequence created a spurious old positive numerator.
        _,old_false,_=self.run_header([1.,.8523714542388916,.23590151965618134,.20000000298023224])
        self.assertGreater(old_false[4],0.0); self.assertEqual(old_false[0],0.0)
        # A real tiny violation was rounded to zero by the old subtraction.
        _,old_missed,_=self.run_header([1.,.875,.25,.1250000149011612,0.])
        self.assertEqual(old_missed[4],0.0); self.assertGreater(old_missed[0],0.0)

    def test_empty_and_one_bin(self):
        for t in ([1,1,1],[1,.5]):
            _,out,g=self.run_header(t);self.assertEqual(out[3],0);self.assertTrue(all(x==0 for x in g))

    def test_random_forward_matches_float64_identity(self):
        rng=random.Random(9741)
        for _ in range(1200):
            n=rng.randint(1,32);cuts=sorted([rng.random() for _ in range(n+1)],reverse=True)
            t,out,_=self.run_header(cuts)
            m=[t[i]-t[i+1] for i in range(n)];v=abs(m[0])+sum(abs(m[i]-m[i-1]) for i in range(1,n))+abs(m[-1]);ref=max(0.,v-2*max(m))
            self.assertAlmostEqual(out[0],ref,delta=2e-6)

    def test_smooth_branch_gradient_central_difference(self):
        t=[1.,.79,.76,.47,.39];eps=.2;_,out,g=self.run_header(t,eps)
        self.assertGreater(out[3],0)
        h=2e-4
        for j in range(len(t)):
            a=t.copy();b=t.copy();a[j]+=h;b[j]-=h
            _,p,_=self.run_header(a,eps);_,m,_=self.run_header(b,eps)
            self.assertAlmostEqual(g[j],(p[3]-m[3])/(2*h),delta=1e-3)

if __name__=='__main__': unittest.main()

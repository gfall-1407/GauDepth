"""Targeted synthetic CUDA checks for K=1,8,32, using isolated extension."""
import math, sys, unittest
from pathlib import Path
try:
 import torch
except ImportError:
 torch = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_unimodal_regularizer as baseline

@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
class StableBinsTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  baseline.CudaSegmentedTests.setUpClass(); cls.mod=baseline.CudaSegmentedTests.mod
 def render(self, bins, enabled, a):
  tangent=math.tan(math.pi/6);proj=torch.zeros((4,4),device='cuda')
  proj[0,0]=proj[1,1]=1/tangent;proj[3,2]=1;proj[2,2]=100/(100-.01);proj[2,3]=-1/(100-.01)
  settings=self.mod.GaussianRasterizationSettings(image_height=64,image_width=64,tanfovx=tangent,tanfovy=tangent,kernel_size=0.,bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),projmatrix=proj.T.contiguous(),sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,require_depth=True,debug=True,unimodal_enabled=enabled,unimodal_min_mass=.5,unimodal_bins=bins,unimodal_near=0.,unimodal_far=4.)
  means=torch.tensor([[.01,.015,.6],[-.01,.01,1.6],[.015,-.01,2.6],[-.01,-.015,3.6]],device='cuda')
  return self.mod.GaussianRasterizer(settings)(means3D=means,means2D=torch.zeros_like(means),opacities=a,colors_precomp=torch.full((4,3),.3,device='cuda'),scales=torch.tensor([[.12,.10,.05]]*4,device='cuda'),rotations=torch.tensor([[1.,0.,0.,0.]]*4,device='cuda'))
 def test_bins_and_gradients(self):
  for bins in (1,8,32):
   with self.subTest(bins=bins):
    a=torch.tensor([[.26],[.08/.74],[.32/.66],[.04/.34]],device='cuda',requires_grad=True)
    off=self.render(bins,False,a);on=self.render(bins,True,a)
    for x,y in zip(off,on[:6]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    loss=on[6][:,30:34,30:34].sum();self.assertTrue(bool(on[7][:,30:34,30:34].all()))
    loss.backward();self.assertTrue(bool(torch.isfinite(a.grad).all()))
    if bins==1:
     self.assertEqual(loss.item(),0.);self.assertEqual(a.grad.abs().max().item(),0.)
    else:
     self.assertGreater(loss.item(),0.)
     num=baseline.finite_diff(lambda v:self.render(bins,True,torch.tensor(v,device='cuda').reshape(4,1))[6][:,30:34,30:34].sum().item(),a.detach().flatten().tolist(),2e-3)
     for actual,expected in zip(a.grad.flatten().tolist(),num):self.assertAlmostEqual(actual,expected,delta=4e-3)
if __name__=='__main__':unittest.main()

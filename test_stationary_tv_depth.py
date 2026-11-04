"""CUDA-only numerical acceptance for stationary-TV; no scene or model writes."""
import unittest
import torch
from stationary_tv_depth import stationary_depth


class StationaryTVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')

    def p(self,m,s,a):
        return tuple(torch.tensor([x],device='cuda',dtype=torch.float64,requires_grad=True)
                     for x in (m,s,a))

    def test_analytic_front_layer_gradient(self):
        p=self.p([-3.,3.],[.2,.2],[.4,.9]);m,s,a=p
        exact=m[:,1]-s[:,1]*torch.sqrt(2*torch.log(a[:,1]/(1-(.5/(1-a[:,0])).square())))
        expected=torch.autograd.grad(exact.sum(),p)
        r=stationary_depth(*p,near=-10,far=10)
        self.assertTrue(bool(r.valid))
        actual=torch.autograd.grad(r.depth.sum(),p)
        self.assertLess(float((r.depth-exact).detach().abs()),1e-9)
        for u,v in zip(actual,expected): self.assertLess(float((u-v).abs().max()),1e-8)

    def test_gradcheck_independent_forward(self):
        p=self.p([-.31,.27],[.91,1.03],[.57,.52])
        fn=lambda *x:stationary_depth(*x,near=-10,far=10).depth
        self.assertTrue(bool(stationary_depth(*p,near=-10,far=10).valid))
        self.assertGreater(float(torch.autograd.grad(fn(*p).sum(),p[0])[0].abs().sum()),.1)
        self.assertTrue(torch.autograd.gradcheck(fn,p,eps=1e-6,atol=2e-6,rtol=2e-5))

    def test_refinement_branch_and_gradients(self):
        p=self.p([-.5,.1,1.2],[.7,.4,.6],[.5,.7,.6])
        out=[]
        for n in (1,2,4):
            r=stationary_depth(*p,near=-10,far=10,subdivisions=n)
            out.append((r,torch.autograd.grad(r.depth.sum(),p)))
        for (a,ga),(b,gb) in zip(out,out[1:]):
            self.assertTrue(bool(a.valid & b.valid))
            self.assertEqual(a.prefix_count,b.prefix_count)
            self.assertEqual(len(a.critical_points),len(b.critical_points))
            self.assertLess(float((a.depth-b.depth).detach().abs()),1e-9)
            for u,v in zip(ga,gb): self.assertLess(float((u-v).abs().max()),1e-8)

    def test_finite_near_derivative(self):
        p=self.p([.2],[1.],[.9])
        fn=lambda *x:stationary_depth(*x,near=-3,far=10,max_near_potential=.01).depth
        r=stationary_depth(*p,near=-3,far=10,max_near_potential=.01)
        self.assertTrue(bool(r.valid))
        self.assertGreater(float(r.near_potential),.001)
        self.assertTrue(torch.autograd.gradcheck(fn,p,eps=1e-6,atol=2e-6,rtol=2e-5))

    def test_no_root_and_degenerate_root(self):
        for p in (self.p([-.3,.3],[1.,1.],[.3,.3]),self.p([0.],[1.],[.75])):
            r=stationary_depth(*p,near=-10,far=10)
            self.assertFalse(bool(r.valid))
            g=torch.autograd.grad(r.depth.sum(),p)
            self.assertTrue(all(bool((x==0).all()) for x in g))


if __name__=='__main__': unittest.main(verbosity=2)

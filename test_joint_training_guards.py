"""CUDA-only regressions for preserved original signed-Inf opacity logits."""
from types import SimpleNamespace
import unittest
import torch
from train_joint_matched import (PARAMETER_NAMES,GuardFailure,
    initialize_parameter_policy,validate_parameter_policy)


class TrainingParameterPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')

    def model(self):
        g=SimpleNamespace(_appearance_embeddings=None,optimizer=None)
        shapes={'_xyz':(3,3),'_features_dc':(3,1,3),'_features_rest':(3,15,3),
                '_scaling':(3,3),'_rotation':(3,4)}
        for name,shape in shapes.items():
            setattr(g,name,torch.nn.Parameter(torch.ones(shape,device='cuda')))
        g._opacity=torch.nn.Parameter(torch.tensor([[float('inf')],[float('-inf')],[0.]],device='cuda'))
        g.filter_3D=torch.ones((3,1),device='cuda')
        g.optimizer=torch.optim.Adam([dict(params=[getattr(g,name)],name=name,lr=.001)
                                      for name in PARAMETER_NAMES],eps=1e-15)
        return g

    def step(self,g):
        loss=g._opacity.sigmoid().sum()+sum(getattr(g,name).square().sum()
                                             for name in PARAMETER_NAMES if name!='_opacity')
        loss.backward();validate_parameter_policy(g,'pre_step',require_zero_gradient=True)
        g.optimizer.step();return validate_parameter_policy(g,'post_step',require_zero_gradient=True)

    def test_signed_inf_retained_geometry_still_updates(self):
        g=self.model();initialize_parameter_policy(g);before=g._xyz.detach().clone()
        health=self.step(g)
        self.assertEqual(g._initial_parameter_state['opacity']['positive_inf'],1)
        self.assertEqual(g._initial_parameter_state['opacity']['negative_inf'],1)
        self.assertTrue(torch.isposinf(g._opacity[0]));self.assertTrue(torch.isneginf(g._opacity[1]))
        self.assertFalse(bool((before==g._xyz).all()))
        self.assertTrue(health['saturated_opacity_gradient_zero'])
        self.assertTrue(health['saturated_opacity_moments_zero'])

    def test_original_opacity_nan_rejected(self):
        g=self.model()
        with torch.no_grad():g._opacity[2]=float('nan')
        with self.assertRaises(GuardFailure):initialize_parameter_policy(g)

    def test_other_original_parameter_inf_rejected(self):
        g=self.model()
        with torch.no_grad():g._xyz[0,0]=float('inf')
        with self.assertRaises(GuardFailure):initialize_parameter_policy(g)

    def test_new_inf_rejected(self):
        g=self.model();initialize_parameter_policy(g)
        with torch.no_grad():g._opacity[2]=float('inf')
        with self.assertRaises(GuardFailure):validate_parameter_policy(g,'new_inf')

    def test_original_sign_change_rejected(self):
        g=self.model();initialize_parameter_policy(g)
        with torch.no_grad():g._opacity[0]=float('-inf')
        with self.assertRaises(GuardFailure):validate_parameter_policy(g,'sign_flip')

    def test_saturated_nonzero_gradient_rejected(self):
        g=self.model();initialize_parameter_policy(g)
        g._opacity.grad=torch.zeros_like(g._opacity);g._opacity.grad[0]=1.
        with self.assertRaises(GuardFailure):validate_parameter_policy(g,'gradient',require_zero_gradient=True)

    def test_saturated_nonzero_moment_rejected(self):
        g=self.model();initialize_parameter_policy(g);self.step(g)
        g.optimizer.state[g._opacity]['exp_avg'][0]=1.
        with self.assertRaises(GuardFailure):validate_parameter_policy(g,'moment')

    def test_any_nonfinite_adam_state_rejected(self):
        g=self.model();initialize_parameter_policy(g);self.step(g)
        g.optimizer.state[g._features_dc]['exp_avg_sq'][0,0,0]=float('inf')
        with self.assertRaises(GuardFailure):validate_parameter_policy(g,'nonfinite_moment')

    def test_weight_decay_rejected(self):
        g=self.model();g.optimizer.param_groups[0]['weight_decay']=.1
        with self.assertRaises(GuardFailure):initialize_parameter_policy(g)


if __name__=='__main__':unittest.main(verbosity=2)

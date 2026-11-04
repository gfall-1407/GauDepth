import uuid
import shutil
from contextlib import contextmanager
import unittest
from pathlib import Path
import numpy as np
from collective_readout import reference_opacity, first_crossing_numpy
from scene_compare_io import camera_alignment, read_cfg, stage_model, sha256, validate_profile


@contextmanager
def fixture_directory():
    base = Path(__file__).resolve().parent / '.test-fixtures'
    base.mkdir(exist_ok=True)
    folder = base / uuid.uuid4().hex
    folder.mkdir()
    try:
        yield str(folder)
    finally:
        if folder.resolve().parent != base.resolve():
            raise ValueError('test cleanup escaped fixture root')
        shutil.rmtree(folder)


class SceneCompareTests(unittest.TestCase):
    def test_reviewed_profile_rejects_wrong_model_count_resolution_and_images(self):
        from PIL import Image
        with fixture_directory() as d:
            base=Path(d).resolve();model=base/'model';source=base/'scan40';dtu=base/'gt'
            ply=model/'point_cloud'/'iteration_7000'/'point_cloud.ply'
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nend_header\n0\n1\n')
            (source/'images').mkdir(parents=True);dtu.mkdir()
            im=source/'images'/'0000.png';Image.new('RGB',(4,2)).save(im)
            profile=dict(scene='scan40',iteration=7000,ply_sha256=sha256(ply),gaussians=2,
                image_names=['0000'],raw_dimensions=[4,2],cfg={'resolution':2},
                hashes=[dict(root='source',relative='images/0000.png',sha256=sha256(im))])
            run=lambda p,cfg={'resolution':2}:validate_profile(model,source,dtu,7000,cfg,['0000'],p)
            run(profile)
            for bad in ({**profile,'ply_sha256':'0'*64},{**profile,'gaussians':3},
                        {**profile,'raw_dimensions':[2,4]},{**profile,'image_names':['0040']}):
                with self.assertRaises(ValueError):run(bad)
            with self.assertRaises(ValueError):run(profile,{'resolution':1})
            Image.new('RGB',(4,2),(20,30,40)).save(im)
            with self.assertRaises(ValueError):run(profile)

    def test_single_analytic_root(self):
        fn = lambda z: reference_opacity(z,[2.],[.2],[.8])
        z, valid = first_crossing_numpy(fn,1)
        exact = 2-.2*np.sqrt(2*np.log(.8/.5))
        self.assertTrue(valid[0])
        self.assertLess(abs(z[0]-exact),8/2**22)

    def test_collective_weak_components_can_cross(self):
        fn = lambda z: reference_opacity(z,[2.,2.],[.2,.2],[.4,.4])
        z,valid=first_crossing_numpy(fn,1)
        self.assertTrue(valid[0])
        self.assertLess(abs(fn(z)[0]-.5),2e-6)

    def test_no_root_and_near_clipped_are_not_baseline_fallback(self):
        z,v=first_crossing_numpy(lambda z:reference_opacity(z,[2.],[.1],[.3]),1)
        self.assertFalse(v[0]);self.assertEqual(z[0],0)
        z,v=first_crossing_numpy(lambda z:reference_opacity(z,[.1],[.1],[.8]),1)
        self.assertFalse(v[0]);self.assertEqual(z[0],0)

    def test_permutation_and_monotonicity(self):
        z=np.linspace(.201,8,1000)
        a=reference_opacity(z,[1.,2.,4.],[.1,.4,.2],[.2,.6,.8])
        b=reference_opacity(z,[4.,1.,2.],[.2,.1,.4],[.8,.2,.6])
        np.testing.assert_allclose(a,b,atol=2e-16)
        self.assertTrue((np.diff(a)>=-1e-15).all())

    def test_alignment_camera_only(self):
        p=np.random.default_rng(12).normal(size=(64,3))
        r=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        g=p@r.T*314.2+[20,30,40]
        a=camera_alignment(p,g)
        self.assertAlmostEqual(a['scale_mm'],314.2)
        self.assertLess(a['max_mm'],1e-10)
        with self.assertRaises(ValueError):camera_alignment(p,g[::-1])

    def test_no_cfg_code_execution(self):
        with fixture_directory() as d:
            f=Path(d)/'cfg'
            f.write_text("Namespace(x=__import__('os').getcwd())")
            with self.assertRaises(ValueError):read_cfg(f)

    def test_staging_preserves_model_and_refuses_overwrite(self):
        with fixture_directory() as d:
            root=Path(d);original=root/'original';target=root/'new'
            ply=original/'point_cloud'/'iteration_7000'/'point_cloud.ply'
            ply.parent.mkdir(parents=True);ply.write_bytes(b'original-test-ply')
            (original/'cfg_args').write_text("Namespace(eval=False, resolution=2)")
            before=sha256(ply)
            stage_model(original,target,root/'scan110',7000)
            self.assertEqual(sha256(ply),before)
            self.assertEqual(sha256(target/'point_cloud'/'iteration_7000'/'point_cloud.ply'),before)
            with self.assertRaises(FileExistsError):stage_model(original,target,root/'scan110',7000)


if __name__=='__main__':unittest.main()

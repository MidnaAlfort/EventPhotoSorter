import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageFilter

from photo_sorter.api_verifier import ApiVerifier, _QualityCountSchema, _subject_box
from photo_sorter.catalog import api_reference_catalog, catalog_signature, discover_reference_catalog
from photo_sorter.engine import PhotoSorter
from photo_sorter.models import SorterConfig, LocalCandidate, AvatarCountDecision, VisionDecision
from photo_sorter.quality import assess_quality, QualityScreen
from photo_sorter.reference_learning import add_confirmed_reference
from photo_sorter.regional_policy import RegionalPrepared, scene_consensus, RegionalViews
from photo_sorter.regions import RegionScene, save_reviewed_regions, read_reviewed_regions
from photo_sorter.state_store import save_record, load_records
from test_improvements import make_sort_tree


class LocalPolicyTests(unittest.TestCase):
    def test_confirmed_photo_matches_by_content_after_reference_is_renamed(self):
        with tempfile.TemporaryDirectory() as d:
            root, work, refs, out = make_sort_tree(Path(d),1)
            identity = discover_reference_catalog(refs)[0].identity_id
            add_confirmed_reference(refs,identity,work/'target_0.png',local_only=True)
            sorter=PhotoSorter(SorterConfig(work,refs,out,mode='api',api_key='test',dry_run=True))
            sorter.api_verifier=MagicMock()
            record=sorter.run()[0]
            self.assertEqual((record.route,record.api_calls),(identity,0))
            sorter.api_verifier.verify.assert_not_called()

    def test_two_scale_requires_matching_boxes_not_just_equal_counts(self):
        a = RegionScene([[.1,.1,.4,.8],[.6,.1,.9,.8]], [.8,.8])
        b = RegionScene([[.6,.1,.9,.8],[.1,.1,.4,.8]], [.7,.7])
        self.assertTrue(scene_consensus(a,b))
        self.assertFalse(scene_consensus(a, RegionScene([[.2,.1,.5,.8],[.5,.1,.8,.8]],[.8,.8])))
        self.assertFalse(scene_consensus(a, RegionScene(a.boxes, [.49,.8])))

    def test_zero_cropped_and_dominant_scenes_are_not_auto_counted(self):
        for scene in (RegionScene([], []), RegionScene([[0,0,.5,.8]],[.99]),
                      RegionScene([[.25,.1,.75,.9],[.8,.2,.9,.4]],[.99,.99])):
            self.assertFalse(scene_consensus(scene, scene))

    def test_group_reference_is_not_used_as_background_identity(self):
        detector = MagicMock(namespace='test')
        detector.detect.return_value = RegionScene([[.1,.1,.4,.9],[.6,.1,.9,.9]],[.9,.9])
        with Image.new('RGB',(100,100)) as image:
            self.assertEqual(RegionalViews(detector)(Path('test'),image), [])

    def test_confirmed_local_reference_preserves_api_prefix_and_annotation(self):
        with tempfile.TemporaryDirectory() as d:
            root, work, refs, output = make_sort_tree(Path(d),1)
            identities = discover_reference_catalog(refs)
            before = catalog_signature(api_reference_catalog(identities))
            source = work/'target_0.png'
            save_reviewed_regions(source, [[.1,.1,.9,.9]])
            dest, added = add_confirmed_reference(refs, identities[0].identity_id, source, local_only=True)
            after = discover_reference_catalog(refs)
            self.assertTrue(added)
            self.assertEqual(dest.parent.name, '_local')
            self.assertTrue(read_reviewed_regions(dest).reviewed)
            self.assertEqual(before, catalog_signature(api_reference_catalog(after)))
            self.assertGreater(len(after[0].images),len(identities[0].images))


class QualityTests(unittest.TestCase):
    def test_textured_subject_passes_but_blur_and_flat_material_need_review(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'photo.png'
            scene = RegionScene([[.1,.1,.9,.9]],[.9])
            noise = Image.fromarray(np.random.default_rng(42).integers(0,256,(800,800,3),dtype=np.uint8))
            noise.save(path)
            self.assertEqual(assess_quality(path,scene).status,'sharp')
            noise.filter(ImageFilter.GaussianBlur(12)).save(path)
            self.assertEqual(assess_quality(path,scene).status,'review')
            Image.new('RGB',(800,800),'white').save(path)
            self.assertEqual(assess_quality(path,scene).status,'review')

    def test_background_texture_does_not_rescue_blurry_subject(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'photo.png'
            image = Image.fromarray(np.random.default_rng(1).integers(0,256,(800,800,3),dtype=np.uint8))
            image.paste((128,128,128),(200,80,600,720))
            image.save(path)
            self.assertEqual(assess_quality(path,RegionScene([[.25,.1,.75,.9]],[.9])).status,'review')

    def test_missing_regions_never_count_as_sharp_or_blurry(self):
        self.assertEqual(assess_quality(Path('not-opened'),RegionScene([],[])).status,'review')

    def test_quality_cache_uses_content_boxes_model_and_zero_usage_on_hit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'p.png'; q = root/'q.png'
            Image.new('RGB',(160,160),'red').save(p); q.write_bytes(p.read_bytes())
            api = ApiVerifier('test','gpt-5.6-luna',1024,root/'cache')
            api.client = MagicMock()
            parsed = _QualityCountSchema(visible_avatar_count=1, uncertain=False,
                primary_subject_dominant=False, confidence=.99, reason_short='solo',
                primary_subject_box=[.1,.1,.9,.9], quality_status='blurry',quality_confidence=.99,quality_reason='blur')
            api.client.responses.parse.return_value = SimpleNamespace(output_parsed=parsed,
                usage=SimpleNamespace(input_tokens=100,output_tokens=20,input_tokens_details=None))
            _, first = api.count_and_quality(p,[[.1,.1,.9,.9]])
            _, cached = api.count_and_quality(q,[[.1,.1,.9,.9]])
            self.assertEqual(first.input_tokens,100)
            self.assertEqual(cached.input_tokens,0)
            self.assertTrue(cached.api_cache_hit)
            self.assertEqual(cached.primary_subject_box,[.1,.1,.9,.9])
            api.client.responses.parse.assert_called_once()
            api.count_and_quality(p,[])
            self.assertEqual(api.client.responses.parse.call_count,2)

    def test_invalid_or_group_subject_boxes_are_not_trusted(self):
        for box, count in (([.9,0,.1,1],1),([0,0,2,1],1),([0,0,1,1],2),([0,0,float('nan'),1],1)):
            self.assertIsNone(_subject_box(SimpleNamespace(primary_subject_box=box,visible_avatar_count=count)))


class RoutingTests(unittest.TestCase):
    def setup_case(self, root, *, scene=None, quality=None, confirmed=False, weak=False, dry_run=True):
        _, work, refs, out = make_sort_tree(root,1)
        ids = discover_reference_catalog(refs)
        sorter = PhotoSorter(SorterConfig(work,refs,out,mode='regional',api_key='test',dry_run=dry_run,
                            regional_local_first=True,quality_filter=quality is not None))
        scene = scene or RegionScene([[.1,.1,.9,.9]],[.9])
        candidates = [LocalCandidate(ids[0].identity_id,.9 if not weak else .6),LocalCandidate('other',.65)]
        sorter._regional_matcher = MagicMock()
        sorter._regional_matcher.prepare.return_value = RegionalPrepared(scene,candidates,not weak,confirmed,'test',quality)
        api = sorter.api_verifier = MagicMock()
        api.count.return_value = AvatarCountDecision(1,False,False,.99,'solo')
        api.verify.return_value = VisionDecision(1,[ids[0].identity_id],.99,False,'matched')
        return sorter, api, ids[0].identity_id, work/'target_0.png'

    def test_confirmed_solo_skips_all_api(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, api, identity, source = self.setup_case(Path(d),confirmed=True)
            record = sorter.run()[0]
            self.assertEqual((record.route,record.api_calls),(identity,0))
            api.count.assert_not_called(); api.verify.assert_not_called()

    def test_weak_solo_uses_one_joint_request(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, api, identity, source = self.setup_case(Path(d),weak=True)
            record = sorter.run()[0]
            self.assertEqual((record.api_calls,record.screen_calls),(1,0))
            api.count.assert_not_called(); api.verify.assert_called_once()

    def test_count_disagreement_overrides_local_identity(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, api, identity, source = self.setup_case(Path(d))
            api.count.return_value = AvatarCountDecision(4,False,False,.99,'crowd')
            self.assertEqual(sorter.run()[0].route,'大勢')
            api.verify.assert_not_called()

    def test_blur_isolated_before_identity_and_count_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, api, identity, source = self.setup_case(Path(d),quality=QualityScreen('review','suspect'),dry_run=False)
            api.count_and_quality.return_value = (SimpleNamespace(quality_status='blurry',quality_confidence=.99,quality_reason='defocused'),
                AvatarCountDecision(1,False,False,.99,'solo',input_tokens=100))
            record = sorter.run()[0]
            self.assertEqual((record.route,record.api_calls,record.quality_calls),('ピンぼけ',1,1))
            self.assertEqual(record.input_tokens,100)
            self.assertFalse(source.exists())
            self.assertTrue(Path(record.destination).is_file())
            api.count.assert_not_called(); api.verify.assert_not_called()

    def test_uncertain_quality_or_api_disabled_goes_to_quality_review(self):
        for enabled in (True,False):
            with tempfile.TemporaryDirectory() as d:
                sorter, api, identity, source = self.setup_case(Path(d),quality=QualityScreen('review','suspect'))
                sorter.config.quality_api_review = enabled
                api.count_and_quality.return_value = (SimpleNamespace(quality_status='blurry',quality_confidence=.7,quality_reason='uncertain'),
                    AvatarCountDecision(1,False,False,.99,'solo'))
                record = sorter.run()[0]
                self.assertEqual(record.route,'品質要確認')
                self.assertEqual(record.api_calls,int(enabled))
                self.assertTrue(source.exists())
                api.verify.assert_not_called()

    def test_quality_pass_reuses_its_count_for_local_identity(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, api, identity, source = self.setup_case(Path(d),quality=QualityScreen('review','suspect'))
            api.count_and_quality.return_value = (SimpleNamespace(quality_status='sharp',quality_confidence=.99,quality_reason='sharp'),
                AvatarCountDecision(1,False,False,.99,'solo'))
            record = sorter.run()[0]
            self.assertEqual((record.route,record.api_calls),(identity,1))
            api.count.assert_not_called(); api.verify.assert_not_called()

    def test_api_foreground_box_enables_local_identity_despite_screen_detections(self):
        with tempfile.TemporaryDirectory() as d:
            scene=RegionScene([[.1,.1,.4,.9],[.6,.1,.9,.9]],[.9,.9])
            sorter, api, identity, source = self.setup_case(Path(d),scene=scene,weak=True)
            api.count.return_value=AvatarCountDecision(1,False,False,.99,'foreground',primary_subject_box=[.1,.1,.4,.9])
            sorter._regional_matcher.matcher.rank_box.return_value=([LocalCandidate(identity,.9),LocalCandidate('other',.65)],True)
            record=sorter.run()[0]
            self.assertEqual((record.route,record.api_calls),(identity,1))
            api.verify.assert_not_called()

    def test_failed_quality_keeps_original_and_checkpoint_is_readable(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, api, identity, source=self.setup_case(Path(d),quality=QualityScreen('review','suspect'),dry_run=False)
            api.count_and_quality.side_effect=RuntimeError('offline')
            sorter.config.save_records = True
            record=sorter.run()[0]
            self.assertEqual(record.route,'エラー')
            self.assertTrue(source.is_file())
            self.assertEqual(len(load_records(sorter.config.output_dir/'.photo_sorter_state.sqlite3')),1)


class CheckpointTests(unittest.TestCase):
    def test_incremental_records_merge_and_replace_without_losing_other_photos(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.sqlite3'
            save_record(path,'a',{'route':'要確認'})
            save_record(path,'b',{'route':'大勢'})
            save_record(path,'a',{'route':'person'})
            self.assertEqual(load_records(path),{'a':{'route':'person'},'b':{'route':'大勢'}})


if __name__ == '__main__':
    unittest.main()

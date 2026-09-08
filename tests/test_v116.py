import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from photo_sorter.burst import describe, similar
from photo_sorter.catalog import discover_reference_catalog
from photo_sorter.cli import build_parser
from photo_sorter.engine import PhotoSorter
from photo_sorter.models import SorterConfig, AvatarCountDecision, VisionDecision, LocalCandidate
from photo_sorter.quality import QualityScreen
from photo_sorter.regional_policy import RegionalPrepared
from photo_sorter.regions import RegionScene
from photo_sorter.reference_suggestions import suggest_references
from photo_sorter.reference_learning import add_confirmed_reference
from test_sorter import make_sort_tree


class CacheSeparationTests(unittest.TestCase):
    def test_reference_cache_survives_restart_but_work_photo_always_calls_api(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, refs, out = make_sort_tree(Path(d), 1)
            config = SorterConfig(work, refs, out, api_key='test', cache_dir=Path(d)/'cache', reuse_api_results=True)
            image = work/'target_0.png'
            first = PhotoSorter(config)
            count = AvatarCountDecision(1, False, False, .99, 'solo', primary_subject_box=[.1,.1,.9,.9])
            reference_api = first._reference_api()
            reference_api._count_uncached = MagicMock(return_value=count)
            self.assertFalse(reference_api.count(image).api_cache_hit)
            second = PhotoSorter(config)
            second._reference_api()._count_uncached = MagicMock(side_effect=AssertionError('reference was not cached'))
            self.assertTrue(second._reference_api().count(image).api_cache_hit)
            target_api = second._api()
            self.assertIsNone(target_api._decision_cache)
            target_api._count_uncached = MagicMock(return_value=count)
            target_api.count(image); target_api.count(image)
            self.assertEqual(target_api._count_uncached.call_count, 2)
            self.assertFalse(config.resume)

    def test_changed_reference_or_disabled_cache_calls_again(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, refs, out = make_sort_tree(Path(d), 1)
            image = work/'target_0.png'
            config = SorterConfig(work,refs,out,api_key='test',cache_dir=Path(d)/'cache')
            api = PhotoSorter(config)._reference_api()
            api._count_uncached = MagicMock(return_value=AvatarCountDecision(1,False,False,.99,'solo'))
            api.count(image)
            Image.new('RGB',(32,32),'green').save(image)
            api.count(image)
            self.assertEqual(api._count_uncached.call_count, 2)
            self.assertIsNone(PhotoSorter(replace(config,reuse_reference_regions=False))._reference_api()._decision_cache)

    def test_same_photo_reinserted_is_not_skipped_by_old_history(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, refs, out = make_sort_tree(Path(d), 1)
            p = work/'target_0.png'; original = p.read_bytes()
            identity = discover_reference_catalog(refs)[0].identity_id
            for _ in range(2):
                p.write_bytes(original)
                sorter = PhotoSorter(SorterConfig(work,refs,out,mode='api',api_key='test',save_records=True))
                sorter._load_state = MagicMock(side_effect=AssertionError('history read'))
                sorter.api_verifier = MagicMock()
                sorter.api_verifier.verify.return_value = VisionDecision(1,[identity],.99,False,'solo')
                self.assertEqual(len(sorter.run()),1)
                sorter.api_verifier.verify.assert_called_once()
                self.assertFalse(p.exists())
            # The move helper deduplicates identical output bytes after reclassification.
            self.assertEqual(len(list((out/identity).glob('*.png'))),1)
            self.assertEqual(len(list((out/'処理記録').glob('*.csv'))),2)

    def test_cli_defaults_do_not_resume_and_reference_cache_is_independent(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.resume)
        self.assertTrue(args.reuse_reference_regions)
        self.assertTrue(args.share_burst_identity)


class BurstTests(unittest.TestCase):
    def case(self, root, *, concurrency=6, change=False, quality=False):
        _, work, refs, out = make_sort_tree(root, 0)
        identity = discover_reference_catalog(refs)[0].identity_id
        pixels = np.random.default_rng(8).integers(0,256,(256,256,3),dtype=np.uint8)
        paths=[]
        for index in range(3):
            p=work/f'VRChat_2026-09-06_12-00-0{index}.000_256x256.png'
            adjusted=pixels.copy()
            adjusted[0,0]=index
            if change and index==1:
                adjusted[80:160,80:160]=255
            Image.fromarray(adjusted).save(p); paths.append(p)
        candidates=[LocalCandidate(identity,.74),LocalCandidate('other',.65)]
        def prepare(p):
            q=QualityScreen('review' if quality and p==paths[1] else 'sharp','test')
            return RegionalPrepared(RegionScene([[.1,.1,.9,.9]],[.9]),list(candidates),True,False,'test',q)
        sorter=PhotoSorter(SorterConfig(work,refs,out,mode='regional',api_key='test',api_concurrency=concurrency,
            regional_local_first=True,quality_filter=True))
        sorter._regional_matcher=MagicMock()
        sorter._regional_matcher.prepare.side_effect=prepare
        sorter._regional_matcher.matcher.rank_box.return_value=(candidates,True)
        api=sorter.api_verifier=MagicMock()
        api.count.return_value=AvatarCountDecision(1,False,False,.99,'solo',primary_subject_box=[.1,.1,.9,.9])
        api.verify.return_value=VisionDecision(1,[identity],.99,False,'solo')
        return sorter,api,paths,identity,prepare

    def test_shared_identity_keeps_individual_counts_and_no_transitive_anchor(self):
        for concurrency in (1,6):
            with tempfile.TemporaryDirectory() as d:
                sorter,api,paths,identity,_=self.case(Path(d),concurrency=concurrency)
                sorter.config.save_records = True
                records=sorted(sorter.run(),key=lambda r:r.source)
                self.assertEqual([r.route for r in records],[identity]*3)
                self.assertEqual(api.verify.call_count,1)
                self.assertEqual(api.count.call_count,2)
                self.assertEqual([r.burst_shared_from for r in records],['',str(paths[0]),str(paths[0])])
                self.assertTrue(all(not p.exists() for p in paths))
                summary=json.loads(sorter.manifest_path.with_suffix('.summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['burst_shared_images'],2)

    def test_changed_subject_not_shared_even_when_local_candidate_is_same(self):
        with tempfile.TemporaryDirectory() as d:
            sorter,api,paths,_,_=self.case(Path(d),change=True)
            records=sorter.run()
            second=next(r for r in records if r.source==str(paths[1]))
            self.assertFalse(second.burst_shared_from)
            self.assertEqual(api.verify.call_count,2)

    def test_new_person_count_prevents_identity_share(self):
        with tempfile.TemporaryDirectory() as d:
            sorter,api,paths,_,_=self.case(Path(d))
            api.count.return_value=AvatarCountDecision(2,False,False,.99,'two')
            records=sorter.run()
            self.assertEqual(sum(r.route=='複数人' for r in records),2)
            self.assertFalse(any(r.burst_shared_from for r in records))

    def test_detector_extra_regions_do_not_override_api_foreground_identity(self):
        with tempfile.TemporaryDirectory() as d:
            sorter,api,paths,identity,prepare=self.case(Path(d))
            def group(p):
                result=prepare(p)
                result.scene=RegionScene([[.05,.1,.5,.9],[.55,.1,.98,.9]],[.9,.9])
                return result
            sorter._regional_matcher.prepare.side_effect=group
            api.count.return_value=AvatarCountDecision(2,False,True,.99,'dominant')
            records=sorter.run()
            self.assertTrue(all(r.route==identity for r in records))
            self.assertEqual(api.verify.call_count,3)

    def test_blur_always_checked_and_isolated_before_share(self):
        with tempfile.TemporaryDirectory() as d:
            sorter,api,paths,_,_=self.case(Path(d),quality=True)
            api.count_and_quality.return_value=(SimpleNamespace(quality_status='blurry',quality_confidence=.99,quality_reason='blur'),
                AvatarCountDecision(1,False,False,.99,'solo'))
            records=sorter.run()
            second=next(r for r in records if r.source==str(paths[1]))
            self.assertEqual(second.route,'ピンぼけ')
            self.assertFalse(second.burst_shared_from)
            api.count_and_quality.assert_called_once()

    def test_failed_anchor_falls_back_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            sorter,api,paths,identity,_=self.case(Path(d))
            def verify(p,*args,**kwargs):
                if p==paths[0]: raise RuntimeError('offline')
                return VisionDecision(1,[identity],.99,False,'solo')
            api.verify.side_effect=verify
            records=sorter.run()
            self.assertTrue(paths[0].exists())
            self.assertEqual(sum(bool(r.error) for r in records),1)
            self.assertFalse(any(r.burst_shared_from for r in records))

    def test_missing_foreground_or_disabled_sharing_uses_identity_api(self):
        for enabled in (True,False):
            with tempfile.TemporaryDirectory() as d:
                sorter,api,_,_,_=self.case(Path(d))
                sorter.config.share_burst_identity=enabled
                api.count.return_value.primary_subject_box=None
                self.assertFalse(any(r.burst_shared_from for r in sorter.run()))
                self.assertEqual(api.verify.call_count,3)

    def test_time_gap_reversal_and_rival_identity_block_share(self):
        with tempfile.TemporaryDirectory() as d:
            sorter,api,paths,_,prepare=self.case(Path(d))
            a,b=(describe(p,prepare(p)) for p in paths[:2])
            self.assertTrue(similar(a,b))
            from datetime import timedelta
            self.assertFalse(similar(a,replace(b,timestamp=a.timestamp+timedelta(seconds=4))))
            self.assertFalse(similar(b,a))
            self.assertFalse(similar(a,replace(b,identity='different')))


class SuggestionTests(unittest.TestCase):
    def test_candidate_requires_reviewable_solo_sharp_api_result_and_excludes_registered(self):
        with tempfile.TemporaryDirectory() as d:
            root,work,refs,out=make_sort_tree(Path(d),1)
            identity=discover_reference_catalog(refs)[0]
            dest=out/identity.category/identity.name/'photo.png'
            dest.parent.mkdir(parents=True)
            dest.write_bytes((work/'target_0.png').read_bytes())
            manifest=out/'処理記録'/'result.csv'; manifest.parent.mkdir()
            base=dict(route=identity.identity_id,destination=str(dest),quality_status='sharp',
                visible_avatar_count='1',decision_source='api',confidence='.99',error='',local_gate='no_candidates')
            def write(row):
                with manifest.open('w',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=base.keys()); writer.writeheader(); writer.writerow(row)
            write(base)
            items=suggest_references(manifest,refs)
            self.assertEqual(len(items),1)
            self.assertEqual(items[0]['source'],str(dest.resolve()))
            for key,value in [('quality_status','review'),('visible_avatar_count','2'),('decision_source','burst_identity'),('confidence','.7')]:
                write({**base,key:value}); self.assertEqual(suggest_references(manifest,refs),[])
            write(base)
            add_confirmed_reference(refs,identity.identity_id,dest,local_only=True)
            self.assertEqual(suggest_references(manifest,refs),[])


if __name__ == '__main__':
    unittest.main()

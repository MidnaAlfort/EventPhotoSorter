import unittest

from photo_sorter.avatar_scene import SceneAdvice, parse_scene_advice


class SceneOutputTests(unittest.TestCase):
    def test_subject_coordinates(self):
        self.assertEqual(parse_scene_advice('{"kind":"subject","box":[5,20,850,1000]}'),
                         SceneAdvice("subject", (.005, .02, .85, 1.0)))

    def test_clear_group_and_abstention(self):
        for kind in ("crowd", "uncertain"):
            self.assertEqual(parse_scene_advice('{"kind":"' + kind + '","box":null}'), SceneAdvice(kind))

    def test_invalid_advice_is_rejected(self):
        for answer in ('null', '{}', '{"kind":"person"}', 'maybe a crowd',
                       '{"kind":"crowd","box":[1,2,3,4]}', '{"kind":"subject","box":[]}',
                       '{"kind":"subject","box":[0,true,40,50]}', '{"kind":"subject","box":[0,0,3000,50]}',
                       '{"kind":"subject","box":[300,0,100,20]}'):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_scene_advice(answer))

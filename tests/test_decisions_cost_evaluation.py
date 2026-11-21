import unittest

from tools.evaluate_decisions_cost import evaluate_pair


class DecisionsCostEvaluationTests(unittest.TestCase):
    def test_acceptance_requires_complete_usage_speed_cost_and_quality(self):
        base = {"complete": True, "count": 100, "estimated_jpy": 8.0, "seconds": 100.,
                "wrong_person_routes": [["a.png", "生徒/01"]], "misroutes": 3, "review": 2,
                "errors": 0, "decisions_image_data_url_bytes": 1000}
        improved = {**base, "estimated_jpy": 5.0, "seconds": 105., "misroutes": 2}
        self.assertTrue(evaluate_pair(base, improved)["adopt"])
        for changes in (
            {"complete": False}, {"count": 99}, {"estimated_jpy": None}, {"estimated_jpy": 7.21},
            {"seconds": 111.}, {"wrong_person_routes": [["a.png", "生徒/02"]]},
            {"misroutes": 4}, {"review": 3}, {"errors": 1}, {"decisions_image_data_url_bytes": 999},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(evaluate_pair(base, {**improved, **changes})["adopt"])

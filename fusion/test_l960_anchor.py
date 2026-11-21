import unittest

from l960_anchor import additions


def det(form, box=(0, 0, 100, 100), score=.6):
    return {"form": form, "box": box, "score": score}


class AnchorTest(unittest.TestCase):
    def test_only_missing_nearby_form(self):
        anchor = [det("solid")]
        bulk = det("bulk")
        candidates = [det("solid"), det("bulk", (110, 0, 210, 100)),
                      det("bulk", score=.02), bulk, det("bulk")]
        self.assertEqual(list(additions(anchor, candidates)), [(bulk, 0)])
        self.assertEqual(anchor, [det("solid")])
        self.assertEqual(list(additions([], candidates)), [])

    def test_attributes_come_from_closest_anchor(self):
        anchor = [det("solid", (20, 0, 120, 100)), det("solid")]
        bulk = det("bulk")
        self.assertEqual(list(additions(anchor, [bulk])), [(bulk, 1)])

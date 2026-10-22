import unittest

from src.model.inference.post_process import ActionSpaceSpec, decode_tuple_action, encode_tuple_action


class AblationActionMaskTest(unittest.TestCase):
    def test_flat_action_round_trip_is_bijective(self) -> None:
        spec = ActionSpaceSpec(5, 3, 2)
        for action in range(spec.num_actions_per_rh):
            self.assertEqual(encode_tuple_action(*decode_tuple_action(action, spec), spec), action)


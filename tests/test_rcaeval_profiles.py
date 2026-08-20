"""Regression tests for the RCAEval dataset profiles used by the RE2-TT extension.

The RE2-OB manifest under artifacts/p1/manifests/re2ob/ is frozen, and every
recorded number is keyed by its opaque case IDs. Parameterizing the adapter for
RE2-TT therefore has to be provably identity-preserving, which is what the golden
vectors below pin: they were read out of the frozen sidecar, so any change to the
namespace string, the prefix, or the digest slice fails here instead of silently
renaming 90 already-published cases.
"""

from pathlib import Path
import tempfile
import unittest

from src.data.rcaeval import (
    DATASET_PROFILES,
    RE2OB_PROFILE,
    RE2TT_PROFILE,
    _opaque_case_id,
    load_rcaeval_cases,
    load_re2ob_cases,
    load_re2tt_cases,
)

from scripts.prepare_ext_re2tt_manifests import (
    ExtensionIsolationError,
    _assert_isolated,
)


FROZEN_RE2OB_CASE_IDS = {
    "checkoutservice_cpu/1": "re2ob-9e575f44de987a1a",
    "checkoutservice_cpu/2": "re2ob-505f802370635ec3",
}

RECORDED_RE2TT_CASE_IDS = {
    "ts-auth-service_cpu/1": "re2tt-e44ccb97e7cc1570",
    "ts-auth-service_cpu/2": "re2tt-a0c0a90c5448e7fa",
}


class DatasetProfileTest(unittest.TestCase):
    def test_re2ob_identity_fields_are_frozen(self):
        self.assertEqual(RE2OB_PROFILE.dataset, "RCAEval-RE2-OB")
        self.assertEqual(RE2OB_PROFILE.id_namespace, "RCAEval:RE2-OB")
        self.assertEqual(RE2OB_PROFILE.id_prefix, "re2ob")
        self.assertEqual(RE2OB_PROFILE.uri_namespace, "re2-ob")
        self.assertEqual(RE2OB_PROFILE.replicates, ("1", "2", "3"))

    def test_re2tt_identity_fields_are_frozen(self):
        self.assertEqual(RE2TT_PROFILE.dataset, "RCAEval-RE2-TT")
        self.assertEqual(RE2TT_PROFILE.id_namespace, "RCAEval:RE2-TT")
        self.assertEqual(RE2TT_PROFILE.id_prefix, "re2tt")
        self.assertEqual(RE2TT_PROFILE.uri_namespace, "re2-tt")
        self.assertEqual(RE2TT_PROFILE.replicates, ("1", "2", "3"))

    def test_both_profiles_share_the_auxiliary_entity_filter(self):
        self.assertEqual(
            RE2OB_PROFILE.auxiliary_entities, RE2TT_PROFILE.auxiliary_entities
        )
        self.assertIn("istio-init", RE2TT_PROFILE.auxiliary_entities)

    def test_profiles_are_registered_by_key(self):
        self.assertEqual(DATASET_PROFILES["re2ob"], RE2OB_PROFILE)
        self.assertEqual(DATASET_PROFILES["re2tt"], RE2TT_PROFILE)

    def test_frozen_re2ob_case_ids_are_reproduced(self):
        for relative, expected in sorted(FROZEN_RE2OB_CASE_IDS.items()):
            self.assertEqual(_opaque_case_id(RE2OB_PROFILE, relative), expected)

    def test_recorded_re2tt_case_ids_are_reproduced(self):
        for relative, expected in sorted(RECORDED_RE2TT_CASE_IDS.items()):
            self.assertEqual(_opaque_case_id(RE2TT_PROFILE, relative), expected)

    def test_namespaces_cannot_collide_across_releases(self):
        relative = "shared_cpu/1"
        self.assertNotEqual(
            _opaque_case_id(RE2OB_PROFILE, relative),
            _opaque_case_id(RE2TT_PROFILE, relative),
        )


class RE2TTAdapterTest(unittest.TestCase):
    def make_case(self, root: Path, condition="ts-auth-service_cpu", replicate="1"):
        case_directory = root / condition / replicate
        case_directory.mkdir(parents=True)
        (case_directory / "inject_time.txt").write_text("1705918105", encoding="utf-8")
        (case_directory / "metrics.csv").write_text(
            "time,ts-auth-service_container-cpu\n1,0\n", encoding="utf-8"
        )
        (case_directory / "simple_metrics.csv").write_text(
            "time,ts-auth-service_cpu,ts-auth-service_mem,"
            "ts-auth-mongo_cpu,ts-auth-mongo_mem,istio-init_cpu\n"
            "1,0,0,0,0,0\n",
            encoding="utf-8",
        )
        (case_directory / "logs.csv").write_text(
            "timestamp,container_name\n1,ts-auth-service\n", encoding="utf-8"
        )
        (case_directory / "traces.csv").write_text(
            "startTime,serviceName\n1,ts-auth-service\n", encoding="utf-8"
        )
        return case_directory

    def test_dataset_identity_and_candidate_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            self.make_case(raw_root)
            result = load_re2tt_cases(str(raw_root))

            self.assertEqual(len(result.inputs), 1)
            case = result.inputs[0]
            self.assertEqual(case.dataset, "RCAEval-RE2-TT")
            self.assertTrue(case.case_id.startswith("re2tt-"))
            self.assertTrue(case.metrics.uri.startswith("rcaeval://re2-tt/"))
            # istio-init is filtered; databases stay in the candidate set on purpose,
            # because removing them would require reading the root label.
            self.assertEqual(case.services, ("ts-auth-mongo", "ts-auth-service"))
            self.assertEqual(result.labels[0].root_service, "ts-auth-service")
            self.assertEqual(result.labels[0].fault_type, "cpu")

    def test_hyphenated_service_names_survive_condition_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            self.make_case(raw_root, condition="ts-auth-service_socket")
            result = load_re2tt_cases(str(raw_root))
            self.assertEqual(result.labels[0].root_service, "ts-auth-service")
            self.assertEqual(result.labels[0].fault_type, "socket")

    def test_wrapper_matches_explicit_profile_call(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            self.make_case(raw_root)
            self.assertEqual(
                load_re2tt_cases(str(raw_root)),
                load_rcaeval_cases(str(raw_root), RE2TT_PROFILE),
            )

    def test_same_tree_read_under_both_profiles_yields_disjoint_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            self.make_case(raw_root)
            re2ob_ids = {case.case_id for case in load_re2ob_cases(str(raw_root)).inputs}
            re2tt_ids = {case.case_id for case in load_re2tt_cases(str(raw_root)).inputs}
            self.assertEqual(len(re2ob_ids), 1)
            self.assertEqual(re2ob_ids & re2tt_ids, set())


class ExtensionIsolationTest(unittest.TestCase):
    def test_frozen_subtrees_are_refused(self):
        for protected in ("artifacts/p1", "artifacts/p2", "artifacts/p1/manifests/re2ob"):
            with self.assertRaises(ExtensionIsolationError):
                _assert_isolated(Path(protected))

    def test_extension_root_is_allowed(self):
        _assert_isolated(Path("artifacts/ext/re2tt"))


if __name__ == "__main__":
    unittest.main()

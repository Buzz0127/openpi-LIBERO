from __future__ import annotations

import copy
import unittest

import experiment_identity as identity


IDS = {key: char * 64 for key, char in zip(sorted(identity.IDENTITY_KEYS), "1234")}


class ExperimentIdentityTest(unittest.TestCase):
    def test_real_sha1_git_commit_is_accepted(self) -> None:
        manifest = identity.build_model_manifest(
            model_mode="base",
            openpi_commit="48d1847417356fb38ecb5db45b569f12b2d148e6",
            identities={key: "1" * 64 for key in identity.IDENTITY_KEYS},
            adapter_identity_sha256=None,
            training_seed=None,
            artifact_purpose="test",
        )
        self.assertEqual(manifest["openpi_commit"], "48d1847417356fb38ecb5db45b569f12b2d148e6")

    def test_abbreviated_git_commit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "full lowercase Git object ID"):
            identity.build_model_manifest(
                model_mode="base",
                openpi_commit="48d1847",
                identities={key: "1" * 64 for key in identity.IDENTITY_KEYS},
                adapter_identity_sha256=None,
                training_seed=None,
                artifact_purpose="test",
            )

    def test_base_and_adapter_manifests(self) -> None:
        base = identity.build_model_manifest(model_mode="base", openpi_commit="a" * 64, identities=IDS, adapter_identity_sha256=None, training_seed=None, artifact_purpose="no_gradient_baseline")
        adapter = identity.build_model_manifest(model_mode="base_plus_adapter", openpi_commit="a" * 64, identities=IDS, adapter_identity_sha256="b" * 64, training_seed=42, artifact_purpose="trained_candidate")
        self.assertNotEqual(base["model_identity_sha256"], adapter["model_identity_sha256"])

    def test_identity_drift_fails(self) -> None:
        manifest = identity.build_model_manifest(model_mode="base", openpi_commit="a" * 64, identities=IDS, adapter_identity_sha256=None, training_seed=None, artifact_purpose="baseline")
        tampered = copy.deepcopy(manifest)
        tampered["identities"]["norm_stats_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity hash mismatch"):
            identity.validate_model_manifest(tampered)

    def test_base_rejects_adapter(self) -> None:
        with self.assertRaisesRegex(ValueError, "base mode"):
            identity.build_model_manifest(model_mode="base", openpi_commit="a" * 64, identities=IDS, adapter_identity_sha256="b" * 64, training_seed=None, artifact_purpose="bad")


if __name__ == "__main__":
    unittest.main()

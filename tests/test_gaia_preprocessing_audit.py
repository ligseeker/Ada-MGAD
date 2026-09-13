import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.p5.audit_gaia_multimodal_preprocessing import (
    _insert_parent_rows,
    _open_parent_index,
    _parent_lookup,
)


class TraceAuditParentIndexTests(unittest.TestCase):
    def test_cross_service_key_is_ambiguous_but_same_service_duplicate_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = _open_parent_index(Path(temporary) / "parents.sqlite3")
            key_ambiguous = b"a" * 16
            key_unique = b"b" * 16

            stats = _insert_parent_rows(
                connection,
                [
                    (key_ambiguous, "dbservice1"),
                    (key_ambiguous, "redisservice1"),
                    (key_unique, "webservice1"),
                    (key_unique, "webservice1"),
                ],
            )
            resolved, ambiguous = _parent_lookup(
                connection, [key_ambiguous, key_unique]
            )
            connection.close()

            self.assertEqual(resolved, {key_unique: "webservice1"})
            self.assertEqual(ambiguous, {key_ambiguous})
            self.assertEqual(stats["ambiguous_keys_created"], 1)
            self.assertEqual(stats["same_service_duplicates"], 1)


if __name__ == "__main__":
    unittest.main()

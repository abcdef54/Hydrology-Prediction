"""Rollback-only tests for operational revisions, deletion and source-ID identity."""

from unittest import TestCase
from dotenv import load_dotenv
import sqlalchemy as sa
from src.backend.db.models import Facility, FacilityOperation
from src.backend.db.session import build_engine
from src.backend.ingestion.facilities import normalize_facility, parse_operation, upsert_operations
from src.backend.ingestion.tests.test_facilities import report
from src.backend.ingestion.worker import SourceError


class FacilityDatabaseTests(TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.engine = build_engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.transaction = self.connection.begin()
        self.addCleanup(self.connection.close)
        self.addCleanup(self.transaction.rollback)
        facility = normalize_facility(
            {"id": 990001, "loaicongtrinhid": "3", "tencongtrinh": "Test rollback"}
        )
        self.connection.execute(sa.insert(Facility).values(facility))
        self.row = parse_operation(report(id=990001, tencongtrinh="990001"))

    def stored(self):
        return (
            self.connection.execute(
                sa.select(FacilityOperation.__table__).where(
                    FacilityOperation.source_record_id == 990001
                )
            )
            .mappings()
            .one()
        )

    def test_revision_and_tombstone_preserve_received_at_and_idempotence(self):
        self.assertEqual(upsert_operations(self.connection, [self.row])["inserted"], 1)
        original = self.stored()
        self.assertEqual(upsert_operations(self.connection, [self.row])["unchanged"], 1)
        self.assertEqual(self.stored()["updated_at"], original["updated_at"])
        changed = {**self.row, "source_status": "2", "htl": 99}
        self.assertEqual(
            upsert_operations(self.connection, [changed], snapshot=True)["unchanged"], 1
        )
        self.assertEqual(self.stored()["source_status"], "1")
        self.assertEqual(upsert_operations(self.connection, [changed])["updated"], 1)
        self.assertEqual(self.stored()["source_status"], "2")
        self.assertEqual(self.stored()["received_at"], original["received_at"])
        self.assertEqual(upsert_operations(self.connection, [changed])["unchanged"], 1)

    def test_same_time_distinct_ids_survive_and_reassigned_id_is_rejected(self):
        upsert_operations(self.connection, [self.row, {**self.row, "source_record_id": 990002}])
        self.assertEqual(
            self.connection.scalar(
                sa.select(sa.func.count())
                .select_from(FacilityOperation)
                .where(FacilityOperation.facility_id == "sohoa:990001")
            ),
            2,
        )
        with self.assertRaises(SourceError):
            upsert_operations(self.connection, [{**self.row, "facility_id": "sohoa:990002"}])

    def test_invalid_time_is_stored_with_null_timestamp(self):
        row = parse_operation(report(id=990001, tencongtrinh="990001", gio=""))
        upsert_operations(self.connection, [row])
        self.assertIsNone(self.stored()["observed_at"])
        self.assertEqual(self.stored()["time_status"], "invalid")

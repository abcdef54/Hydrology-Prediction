"""Metadata import contracts without source credentials or database writes."""

from unittest import TestCase
from unittest.mock import MagicMock

from src.backend.ingestion.station_catalog import (
    fetch_station_catalog,
    import_station_catalog,
    normalize_station,
)
from src.backend.ingestion.worker import SourceError


def source_station(code="0074647403"):
    return {
        "code": code,
        "name": "Trạm thử",
        "address": "Huế",
        "geom": {"type": "Point", "coordinates": [107.57, 16.46]},
        "water_station_type": {"name": "flood_3m", "desc": "Tháp báo lũ"},
    }


class StationCatalogTests(TestCase):
    def test_preserves_code_subtype_and_uses_safe_serving_defaults(self):
        water = normalize_station(source_station(), "water_level_station")
        self.assertEqual(water["id"], "0074647403")
        self.assertEqual(water["station_subtype"], "flood_3m")
        self.assertFalse(water["enabled"])
        self.assertFalse(water["prediction_enabled"])
        self.assertEqual(water["measurement_unit"], "mm")
        rain = normalize_station(source_station("750003"), "rain_water_stations")
        self.assertEqual(rain["measurement_unit"], "mm")
        self.assertIsNone(rain["station_subtype"])

    def test_rejects_invalid_coordinates_identity_and_payload(self):
        for invalid in [
            None,
            {},
            source_station(123),
            {**source_station(), "geom": {"type": "Point", "coordinates": [True, 16]}},
            {**source_station(), "geom": {"type": "Point", "coordinates": [107, 95]}},
            {**source_station(), "geom": {"type": "Point", "coordinates": [float("nan"), 16]}},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(SourceError):
                normalize_station(invalid, "water_level_station")

    def test_short_pages_are_not_assumed_to_be_the_last_page(self):
        client = MagicMock()
        client.request.side_effect = [
            {"data": [source_station()]},
            {"data": [source_station("water2")]},
            {"data": []},
            {"data": [source_station("750003")]},
            {"data": []},
        ]
        self.assertEqual(len(fetch_station_catalog(client)), 3)
        self.assertEqual(client.request.call_count, 5)

    def test_duplicate_source_codes_and_bad_arrays_fail_closed(self):
        client = MagicMock()
        client.request.side_effect = [
            {"data": [source_station()]},
            {"data": []},
            {"data": [source_station()]},
        ]
        with self.assertRaises(SourceError):
            fetch_station_catalog(client)
        client.request.side_effect = None
        client.request.return_value = {"data": {}}
        with self.assertRaises(SourceError):
            fetch_station_catalog(client)

    def test_dry_run_does_not_issue_insert_or_update(self):
        station = normalize_station(source_station(), "water_level_station")
        engine = MagicMock()
        connection = engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value = []
        result = import_station_catalog(engine, [station])
        self.assertEqual(result["counts"]["inserted"], 1)
        self.assertEqual(connection.execute.call_count, 2)

    def test_existing_flags_units_alarms_names_are_not_overwritten(self):
        station = normalize_station(source_station(), "water_level_station")
        original = {
            **station,
            "enabled": True,
            "prediction_enabled": True,
            "measurement_unit": "mm",
            "name": "Dã Viên",
            "station_subtype": None,
            "alarm_level_1_m": 1.0,
        }
        engine = MagicMock()
        connection = engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value = [original]
        result = import_station_catalog(engine, [station], dry_run=False)
        self.assertEqual(result["counts"]["updated"], 1)
        update = connection.execute.call_args.args[0]
        self.assertEqual({column.name for column in update._values}, {"station_subtype"})
        connection.execute.reset_mock()
        connection.execute.return_value.mappings.return_value = [
            {**original, "station_subtype": "flood_3m"}
        ]
        repeated = import_station_catalog(engine, [station], dry_run=False)
        self.assertEqual(repeated["counts"]["unchanged"], 1)
        self.assertEqual(connection.execute.call_count, 2)

    def test_source_conflict_is_rejected_before_updates(self):
        station = normalize_station(source_station(), "water_level_station")
        engine = MagicMock()
        connection = engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value = [
            {**station, "source_name": "other"}
        ]
        with self.assertRaises(SourceError):
            import_station_catalog(engine, [station], dry_run=False)

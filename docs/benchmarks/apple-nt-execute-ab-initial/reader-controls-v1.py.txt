"""Corpus, numerical-witness and A/A controls for Apple execute diagnostics."""

import csv
import io
import unittest

import bench_apple_dense_execute as reader


def corpus():
    rows = []
    for shape in reader.SHAPES:
        m, k, n = shape
        count = reader.iterations(shape)
        for retained in (0, 1):
            for round_ in range(reader.ROUNDS):
                for arm in reader.ORDERS[round_ % len(reader.ORDERS)]:
                    duration = count * (200 if arm == 0 else 100)
                    rows.append(
                        list(
                            map(
                                str,
                                [
                                    "ANT_SAMPLE",
                                    1,
                                    m,
                                    k,
                                    n,
                                    retained,
                                    round_,
                                    arm,
                                    count,
                                    duration,
                                    duration,
                                    12345,
                                ],
                            )
                        )
                    )
            rows.append(
                list(
                    map(
                        str,
                        [
                            "ANT_WITNESS",
                            1,
                            m,
                            k,
                            n,
                            retained,
                            reader.ROUNDS * (count + 1),
                            1 if retained else 2 * reader.ROUNDS * (count + 1),
                            0,
                            k * n * 4 if retained else 0,
                        ],
                    )
                )
            )
    return rows


def analyze(rows):
    text = io.StringIO()
    csv.writer(text).writerows(rows)
    text.seek(0)
    return reader.analyze(text)


class ReaderControls(unittest.TestCase):
    def test_positive_complete_corpus(self):
        result = analyze(corpus())
        self.assertEqual((result["sample_count"], result["cell_count"]), (720, 10))
        for cell in result["cells"]:
            for metric in ("cpu", "wall"):
                self.assertTrue(cell[metric]["qualified"])
                self.assertEqual(cell[metric]["legacy_over_native"], 2)

    def test_empty_corpus(self):
        with self.assertRaises(ValueError):
            analyze([])

    def test_missing_sample(self):
        with self.assertRaises(ValueError):
            analyze(corpus()[1:])

    def test_missing_witness(self):
        with self.assertRaises(ValueError):
            analyze(corpus()[:-1])

    def test_duplicate_sample(self):
        rows = corpus()
        rows.insert(1, rows[0])
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_duplicate_witness(self):
        rows = corpus()
        rows.append(rows[-1])
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_wrong_version(self):
        rows = corpus()
        rows[0][1] = "2"
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_unknown_shape_or_retention(self):
        for index, value in [(2, "3"), (3, "0"), (5, "2")]:
            rows = corpus()
            rows[0][index] = value
            with self.assertRaises(ValueError):
                analyze(rows)

    def test_bad_round_arm_or_iterations(self):
        for index, value in [(6, "24"), (6, "-1"), (7, "3"), (8, "1")]:
            rows = corpus()
            rows[0][index] = value
            with self.assertRaises(ValueError):
                analyze(rows)

    def test_invalid_durations(self):
        for index in (9, 10):
            for value in ("0", "-1", "NaN", "inf", str(1 << 64)):
                rows = corpus()
                rows[0][index] = value
                with self.assertRaises(ValueError):
                    analyze(rows)

    def test_checksum_disagreement(self):
        rows = corpus()
        rows[0][11] = "1"
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_wrong_witness(self):
        for index in (6, 7, 8, 9):
            rows = corpus()
            witness = next(row for row in rows if row[0] == "ANT_WITNESS")
            witness[index] = str(int(witness[index]) + 1)
            with self.assertRaises(ValueError):
                analyze(rows)

    def test_wrong_record_kind_or_field_count(self):
        for edit in (lambda row: row.__setitem__(0, "OTHER"), lambda row: row.pop()):
            rows = corpus()
            edit(rows[0])
            with self.assertRaises(ValueError):
                analyze(rows)

    def test_wrong_six_order_sequence(self):
        rows = corpus()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(ValueError):
            analyze(rows)

    def test_blank_record(self):
        with self.assertRaises(ValueError):
            analyze([[]])

    def test_aa_failure_is_retained_and_clock_specific(self):
        rows = corpus()
        for row in rows:
            if (
                row[0] == "ANT_SAMPLE"
                and row[2:6] == ["2", "33", "3", "0"]
                and row[7] == "2"
            ):
                row[9] = str(2 * int(row[9]))
        result = analyze(rows)
        self.assertEqual(result["cell_count"], 10)
        cell = result["cells"][0]
        self.assertFalse(cell["cpu"]["qualified"])
        self.assertEqual(cell["cpu"]["native_aa_ratio"], 2)
        self.assertTrue(cell["wall"]["qualified"])
        self.assertEqual(cell["wall"]["legacy_over_native"], 2)


if __name__ == "__main__":
    unittest.main()

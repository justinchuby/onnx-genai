"""Non-vacuous corpus and A/A qualification controls for the requant reader."""

import csv
import io
import unittest

import bench_qlinear_per_row as reader


def corpus():
    rows = []
    for n in reader.WIDTHS:
        for dtype in reader.DTYPES:
            iterations = 2_000_000 // n
            for repetition in range(reader.ROUNDS):
                for offset in range(3):
                    arm = (repetition + offset) % 3
                    ns_per_element = 1000 if arm == 0 or n < 8 else 500
                    duration = iterations * n * ns_per_element
                    rows.append(
                        list(
                            map(
                                str,
                                [
                                    "QRE_SAMPLE",
                                    1,
                                    n,
                                    dtype,
                                    repetition,
                                    arm,
                                    iterations,
                                    duration,
                                    duration,
                                    12345,
                                ],
                            )
                        )
                    )
            hits = 0 if n < 8 else 2 * reader.ROUNDS * (iterations + 1)
            rows.append(list(map(str, ["QRE_WITNESS", 1, n, dtype, hits])))
    return rows


def analyze(rows):
    data = io.StringIO()
    csv.writer(data).writerows(rows)
    data.seek(0)
    return reader.analyze(data)


class CorpusControls(unittest.TestCase):
    def test_complete_corpus_has_known_ratios(self):
        result = analyze(corpus())
        self.assertEqual(result["sample_count"], 882)
        self.assertEqual(result["cell_count"], 14)
        for cell in result["cells"]:
            for metric in ["cpu", "wall"]:
                self.assertTrue(cell[metric]["qualified"])
                self.assertEqual(
                    cell[metric]["portable_over_native"], 1 if cell["n"] < 8 else 2
                )

    def test_empty_corpus_is_not_success(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze([])

    def test_missing_sample_is_rejected(self):
        rows = corpus()
        rows.pop(0)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze(rows)

    def test_missing_witness_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze(corpus()[:-1])

    def test_duplicate_sample_is_rejected(self):
        rows = corpus()
        rows.append(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(rows)

    def test_duplicate_witness_is_rejected(self):
        rows = corpus()
        rows.append(rows[-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(rows)

    def test_unknown_protocol_is_rejected(self):
        rows = corpus()
        rows[0][1] = "2"
        with self.assertRaisesRegex(ValueError, "protocol"):
            analyze(rows)

    def test_unknown_cell_is_rejected(self):
        rows = corpus()
        rows[0][2] = "10"
        with self.assertRaisesRegex(ValueError, "unknown cell"):
            analyze(rows)

    def test_wrong_iterations_are_rejected(self):
        rows = corpus()
        rows[0][6] = str(int(rows[0][6]) + 1)
        with self.assertRaisesRegex(ValueError, "iteration"):
            analyze(rows)

    def test_invalid_duration_is_rejected(self):
        for field in [7, 8]:
            for value in ["0", "-1", "NaN", "inf", str(1 << 64)]:
                with self.subTest(field=field, value=value):
                    rows = corpus()
                    rows[0][field] = value
                    with self.assertRaises(ValueError):
                        analyze(rows)

    def test_checksum_disagreement_is_rejected(self):
        rows = corpus()
        rows[0][9] = "12346"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            analyze(rows)

    def test_false_native_witness_is_rejected(self):
        rows = corpus()
        row = next(row for row in rows if row[0] == "QRE_WITNESS" and row[2] == "8")
        row[4] = "0"
        with self.assertRaisesRegex(ValueError, "native-call witness"):
            analyze(rows)

    def test_unknown_arm_is_rejected(self):
        rows = corpus()
        rows[0][5] = "3"
        with self.assertRaisesRegex(ValueError, "arm"):
            analyze(rows)

    def test_wrong_field_count_is_rejected(self):
        rows = corpus()
        rows[0].append("extra")
        with self.assertRaisesRegex(ValueError, "field count"):
            analyze(rows)

    def test_wrong_rotation_order_is_rejected(self):
        rows = corpus()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, "rotating arms"):
            analyze(rows)

    def test_bad_aa_cell_is_retained_as_unqualified(self):
        rows = corpus()
        for row in rows:
            if row[0] == "QRE_SAMPLE" and row[2:4] == ["8", "Int8"] and row[5] == "2":
                row[7] = str(2 * int(row[7]))
                row[8] = str(2 * int(row[8]))
        result = analyze(rows)
        self.assertEqual(result["sample_count"], 882)
        self.assertEqual(result["cell_count"], 14)
        cell = next(
            cell
            for cell in result["cells"]
            if cell["n"] == 8 and cell["dtype"] == "Int8"
        )
        for metric in ["cpu", "wall"]:
            self.assertFalse(cell[metric]["qualified"])
            self.assertEqual(cell[metric]["native_aa_ratio"], 2)


if __name__ == "__main__":
    unittest.main()

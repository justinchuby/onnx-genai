#!/usr/bin/env python3
"""Summarize an onnx-tests JUnit report by ONNX operator."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


TEST_NAME = re.compile(r"test_(.+)_([0-9]+)(?:\[|$)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args()

    operators: dict[str, Counter[str]] = defaultdict(Counter)
    for case in ET.parse(args.junit_xml).getroot().iter("testcase"):
        match = TEST_NAME.match(case.attrib["name"])
        if match is None:
            continue
        operator = match.group(1)
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "fail"
        elif case.find("skipped") is not None:
            outcome = "skip"
        else:
            outcome = "pass"
        operators[operator][outcome] += 1

    totals: Counter[str] = sum(operators.values(), Counter())
    print(
        f"{len(operators)} operators; "
        f"{totals['pass']} pass, {totals['fail']} fail, {totals['skip']} skip"
    )
    print("| Operator | Pass | Fail | Skip |")
    print("|---|---:|---:|---:|")
    for operator, outcomes in sorted(operators.items()):
        print(
            f"| {operator} | {outcomes['pass']} | "
            f"{outcomes['fail']} | {outcomes['skip']} |"
        )


if __name__ == "__main__":
    main()

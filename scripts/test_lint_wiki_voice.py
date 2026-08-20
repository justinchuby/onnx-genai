#!/usr/bin/env python3
"""Tests for lint_wiki_voice.py.

The table below is the specification. The rows that must pass are as load
bearing as the rows that must fail: a check that flags ordinary second-person
technical prose gets suppressed or deleted within a week, and then it is
protecting nothing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lint_wiki_voice import check_text

MUST_FAIL = [
    ("an observation attributed to the reader", "你的观察是对的,两者同族。\n"),
    ("a statement attributed to the reader", '你说的"预测第一个字",严格讲是 token。\n'),
    ("a callback to an unseen conversation", "如你所说,这里有坑。\n"),
    ("an earlier exchange in the first person plural", "我们刚才说的那个坑。\n"),
    ("English: a belief attributed to the reader", "Your intuition is right.\n"),
    ("English: a statement attributed to the reader", "What you called the first character.\n"),
    ("English: agreeing with the reader", "You're right, they share a skeleton.\n"),
    ("a heading that quotes the question", '## 三、"模型是不是从模板后面开始预测?"\n'),
    ("an English heading that quotes the question", '## 3. "Does the model predict next?"\n'),
]

MUST_PASS = [
    ("the generic second person addressed to an implementer", "你的转接层需要自己丢掉推理段。\n"),
    ("English generic second person", "Your adapter has to drop the reasoning block.\n"),
    ("a declarative heading", "## 三、生成从模板的末尾开始\n\n正文\n"),
    ("a heading with quotes but no question", '## 五、Muse Glimmer 的"收件人"设计\n\n正文\n'),
    ("a transcript inside a code fence", "正文\n\n```text\nYou asked: what is this?\n```\n"),
    ("a tilde-fenced transcript", "正文\n\n~~~text\nYour observation was noted.\n~~~\n"),
]


class LintWikiVoiceTests(unittest.TestCase):
    def test_passages_that_must_be_flagged(self) -> None:
        for name, text in MUST_FAIL:
            with self.subTest(name):
                self.assertTrue(check_text(text), f"{name} was not flagged")

    def test_passages_that_must_not_be_flagged(self) -> None:
        for name, text in MUST_PASS:
            with self.subTest(name):
                self.assertEqual(check_text(text), [], f"{name} was flagged")

    def test_a_suppressed_range_is_not_flagged(self) -> None:
        text = "<!-- voice-lint: off -->\n你的观察是对的\n<!-- voice-lint: on -->\n"
        self.assertEqual(check_text(text), [])

    def test_suppression_stops_where_it_is_turned_back_on(self) -> None:
        # Otherwise one counterexample table silently exempts the rest of the
        # document, which is how these checks quietly stop applying.
        text = "<!-- voice-lint: off -->\nok\n<!-- voice-lint: on -->\n你的观察是对的\n"
        self.assertTrue(check_text(text))

    def test_the_reported_line_number_is_the_offending_line(self) -> None:
        text = "第一行\n第二行\n你的观察是对的\n"
        self.assertEqual(check_text(text)[0][0], 3)

    def test_an_unclosed_fence_is_reported_rather_than_skipped_silently(self) -> None:
        # A fence opened with ``` is not closed by ~~~, so per CommonMark the
        # block runs to the end of the document and skipping the rest is
        # correct. That is exactly why it has to be reported: otherwise one
        # stray fence exempts the whole file and nothing says so.
        text = "```text\ntranscript\n~~~\n你的观察是对的\n"
        findings = check_text(text)
        self.assertTrue(findings)
        self.assertIn("never closed", findings[-1][2])
        self.assertEqual(findings[-1][0], 1)

    def test_a_closed_fence_is_not_reported_as_unclosed(self) -> None:
        self.assertEqual(check_text("```text\ntranscript\n```\n正文\n"), [])

    def test_the_wiki_itself_is_clean(self) -> None:
        wiki = Path(__file__).resolve().parent.parent / "wiki"
        offenders = {
            f"{path}:{number}: {excerpt}"
            for path in wiki.rglob("*.md")
            for number, excerpt, _ in check_text(path.read_text(encoding="utf-8"))
        }
        self.assertEqual(offenders, set())


if __name__ == "__main__":
    unittest.main()

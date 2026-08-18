from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "validate_wikilinks.py"


class ValidateWikilinksTest(unittest.TestCase):
    def run_validator(self, files: dict[str, str]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            for relative, content in files.items():
                path = vault / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            return subprocess.run(
                ["python3", str(VALIDATOR), str(vault)],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_ignores_literal_examples_but_validates_real_links(self) -> None:
        result = self.run_validator(
            {
                "A.md": (
                    "# A\n\n"
                    "`[[not-a-target]]`\n\n"
                    "````markdown\n```\n[[also-not-a-target]]\n```\n````\n\n"
                    "    [[indented-code-example]]\n\n"
                    "`multiline code span\n[[multiline-code-example]]\ncontinues here`\n\n"
                    "\\[[escaped-example]]\n\n"
                    "- Nested note\n\n"
                    "    [[B]]\n\n"
                    "[[B#Details]]\n"
                ),
                "B.md": "# B\n\n## Details\n",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated 2 wikilink(s)", result.stdout)

    def test_even_backslashes_do_not_escape_wikilink(self) -> None:
        result = self.run_validator({"A.md": "# A\n\n\\\\[[Missing]]\n"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unresolved wikilink [[Missing]]", result.stderr)

    def test_rejects_missing_target(self) -> None:
        result = self.run_validator({"A.md": "# A\n\n[[Missing]]\n"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unresolved wikilink [[Missing]]", result.stderr)

    def test_rejects_missing_fragment(self) -> None:
        result = self.run_validator(
            {
                "A.md": "# A\n\n[[B#Missing section]]\n",
                "B.md": "# B\n\n## Present section\n",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing fragment", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Static regression checks for published notebook decisions and API setup."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_notebook(name: str) -> dict:
    path = PROJECT_ROOT / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


class NotebookDecisionTests(unittest.TestCase):
    def test_sentiment_conclusion_keeps_balanced_linear_svm_as_final_model(self) -> None:
        notebook = read_notebook("02_sentiment_analysis.ipynb")
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

        self.assertIn("Balanced Linear SVM with TF-IDF was selected", markdown)
        self.assertIn("does not provide `predict_proba()` directly", markdown)
        self.assertNotIn("Logistic Regression with TF-IDF was selected as the final", markdown)

    def test_nvidia_environment_imports_precede_key_lookup(self) -> None:
        notebook = read_notebook("04_summarization.ipynb")
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        imports_at = source.index("from dotenv import load_dotenv")
        os_import_at = source.index("import os")
        lookup_at = source.index('nvidia_api_key = os.getenv("NVIDIA_API_KEY")')

        self.assertLess(imports_at, lookup_at)
        self.assertLess(os_import_at, lookup_at)


if __name__ == "__main__":
    unittest.main()

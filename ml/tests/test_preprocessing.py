import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import clean_text, remove_stopwords, preprocess, compute_stats


class TestCleanText(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(clean_text("Python, SQL & NLP!!"), "python sql nlp")

    def test_removes_urls_and_emails(self):
        out = clean_text("Contact me at me@example.com or visit https://site.com")
        self.assertNotIn("@", out)
        self.assertNotIn("http", out)

    def test_collapses_whitespace(self):
        self.assertEqual(clean_text("a    b\n\tc"), "a b c")

    def test_empty_input(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")


class TestStopwords(unittest.TestCase):
    def test_removes_common_stopwords(self):
        out = remove_stopwords("this is a test of the system")
        for sw in ["this", "is", "a", "of", "the"]:
            self.assertNotIn(sw, out.split())
        self.assertIn("test", out.split())
        self.assertIn("system", out.split())

    def test_drops_single_letter_tokens(self):
        out = remove_stopwords("x y machine learning z")
        self.assertNotIn("x", out.split())
        self.assertIn("machine", out.split())


class TestPreprocessPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        out = preprocess("The Machine Learning Engineer used Python, SQL, and TensorFlow.")
        tokens = out.split()
        self.assertIn("machine", tokens)
        self.assertIn("learning", tokens)
        self.assertIn("python", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)


class TestComputeStats(unittest.TestCase):
    def test_counts_on_simple_text(self):
        stats = compute_stats("Hello world. This is DocuMind! Does it work?")
        self.assertEqual(stats["sentence_count"], 3)
        self.assertEqual(stats["word_count"], 8)
        self.assertGreater(stats["char_count"], 0)

    def test_empty_text(self):
        stats = compute_stats("")
        self.assertEqual(stats, {"char_count": 0, "word_count": 0, "sentence_count": 0})

    def test_whitespace_only(self):
        stats = compute_stats("   \n\t  ")
        self.assertEqual(stats["word_count"], 0)


if __name__ == "__main__":
    unittest.main()

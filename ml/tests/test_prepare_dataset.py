import sys
import unittest
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import CATEGORY_MAP, load_and_map


def make_fixture_csv() -> StringIO:
    """Tiny synthetic fixture mimicking the raw source CSV's shape
    (Category, Resume columns; a mix of mapped and unmapped labels).
    NOT real resume data — just enough rows to exercise the filtering
    and remapping logic."""
    rows = [
        ("Data Science", "Experienced with Python, pandas and scikit-learn for predictive modeling."),
        ("Hadoop", "Built ETL jobs on a Hadoop cluster using Hive and Pig."),
        ("ETL Developer", "Designed ETL pipelines moving data from Oracle to a warehouse."),
        ("Java Developer", "Backend development in Java and Spring Boot microservices."),
        ("Python Developer", "Developed REST APIs in Python using Flask and Django."),
        ("DotNet Developer", "Built enterprise applications using C# and the .NET framework."),
        ("Web Designing", "Created responsive websites with HTML, CSS and JavaScript."),
        ("Network Security Engineer", "Configured firewalls and performed vulnerability assessments."),
        ("Database", "Administered MySQL and PostgreSQL databases, wrote backup scripts."),
        ("DevOps Engineer", "Managed CI/CD pipelines with Jenkins and containerized apps."),
        ("Business Analyst", "Gathered requirements and produced process documentation."),
        ("HR", "Managed recruitment, onboarding and employee relations."),
        ("Arts", "Painter and illustrator with gallery exhibitions."),  # unmapped, should be dropped
        ("Sales", "Achieved quarterly sales targets across enterprise accounts."),  # unmapped
        ("Data Science", ""),  # too short, should be dropped
    ]
    csv_text = "Category,Resume\n" + "\n".join(f'"{c}","{r}"' for c, r in rows)
    return StringIO(csv_text)


class TestCategoryMap(unittest.TestCase):
    def test_maps_to_exactly_nine_target_categories(self):
        self.assertEqual(len(set(CATEGORY_MAP.values())), 9)

    def test_no_unintended_categories(self):
        expected = {
            "Data Science", "Data Engineering", "Software Engineering",
            "Web Development", "Cybersecurity", "Database Administration",
            "DevOps / Cloud Computing", "Business Analysis", "Human Resources",
        }
        self.assertEqual(set(CATEGORY_MAP.values()), expected)


class TestLoadAndMap(unittest.TestCase):
    def setUp(self):
        self.df = load_and_map(make_fixture_csv())

    def test_unmapped_categories_are_dropped(self):
        self.assertNotIn("Arts", self.df["Category"].values)
        self.assertNotIn("Sales", self.df["Category"].values)

    def test_short_resumes_are_dropped(self):
        # the empty "Data Science" row should not survive
        self.assertEqual(len(self.df[self.df["Resume"] == ""]), 0)

    def test_categories_are_remapped(self):
        self.assertIn("Software Engineering", self.df["Category"].values)
        self.assertNotIn("Java Developer", self.df["Category"].values)

    def test_merged_categories_present(self):
        # Java + Python + DotNet Developer all collapse into Software Engineering
        se_count = (self.df["Category"] == "Software Engineering").sum()
        self.assertEqual(se_count, 3)
        de_count = (self.df["Category"] == "Data Engineering").sum()
        self.assertEqual(de_count, 2)  # Hadoop + ETL Developer

    def test_raises_on_missing_columns(self):
        bad_csv = StringIO("foo,bar\n1,2")
        with self.assertRaises(ValueError):
            load_and_map(bad_csv)


if __name__ == "__main__":
    unittest.main()

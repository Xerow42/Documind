"""
Pipeline smoke test: proves train_and_evaluate()/save_artifacts() run
end-to-end without errors and produce internally-consistent output.

IMPORTANT: this uses a small SYNTHETIC fixture (hand-written sentences,
not real resumes) purely to exercise the code path. The accuracy/F1
numbers this test sees are meaningless and must never be reported as
DocuMind's real evaluation results — those require training on the
actual Kaggle dataset via `python ml/train.py` (see docs/dataset_strategy.md
and README.md "Known Limitations").
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import build_vectorizer, save_artifacts, train_and_evaluate

CATEGORIES = [
    "Data Science", "Data Engineering", "Software Engineering",
    "Web Development", "Cybersecurity", "Database Administration",
    "DevOps / Cloud Computing", "Business Analysis", "Human Resources",
]

# A handful of distinct, on-topic synthetic sentences per category so the
# TF-IDF vectorizer has *some* separable signal to work with.
SYNTHETIC_SAMPLES = {
    "Data Science": [
        "Built predictive models using Python pandas and scikit-learn for churn analysis",
        "Performed exploratory data analysis and feature engineering on customer datasets",
        "Applied regression and classification algorithms to forecast sales trends",
        "Used Jupyter notebooks to analyze statistical distributions and outliers",
        "Trained machine learning models to predict customer lifetime value",
        "Developed data visualizations to communicate model performance to stakeholders",
    ],
    "Data Engineering": [
        "Designed ETL pipelines to move data from operational databases into a warehouse",
        "Built and scheduled Hadoop batch jobs for large scale data processing",
        "Maintained data pipelines ensuring data quality and consistent schemas",
        "Wrote scripts to extract transform and load data between systems nightly",
        "Optimized data pipeline performance for high volume ingestion",
        "Implemented data validation checks within the ingestion pipeline",
    ],
    "Software Engineering": [
        "Developed backend services in Java using Spring Boot microservices",
        "Built REST APIs in Python with Flask following clean architecture",
        "Wrote unit tests and maintained CI pipelines for a Node.js application",
        "Implemented new features in a .NET enterprise application",
        "Refactored legacy code to improve maintainability and test coverage",
        "Collaborated on code reviews and object oriented software design",
    ],
    "Web Development": [
        "Created responsive websites using HTML CSS and JavaScript",
        "Built single page applications with React and modern component design",
        "Implemented frontend layouts following accessibility best practices",
        "Developed interactive web interfaces with JavaScript and CSS animations",
        "Built and styled landing pages optimized for mobile devices",
        "Integrated frontend components with backend REST APIs",
    ],
    "Cybersecurity": [
        "Configured firewalls and monitored network traffic for intrusions",
        "Performed vulnerability assessments and penetration testing on web applications",
        "Implemented security controls to protect against unauthorized access",
        "Investigated security incidents and produced remediation reports",
        "Managed endpoint security tools and patched known vulnerabilities",
        "Conducted risk assessments and enforced security compliance policies",
    ],
    "Database Administration": [
        "Administered MySQL and PostgreSQL databases including backup strategies",
        "Tuned database queries and indexes to improve performance",
        "Managed database replication and high availability configurations",
        "Wrote stored procedures and maintained relational database schemas",
        "Monitored database health and resolved performance bottlenecks",
        "Performed database migrations and capacity planning",
    ],
    "DevOps / Cloud Computing": [
        "Managed CI CD pipelines using Jenkins for automated deployments",
        "Containerized applications and orchestrated deployments with Kubernetes",
        "Automated infrastructure provisioning using scripts and configuration management",
        "Monitored production systems and maintained deployment automation",
        "Built deployment pipelines integrating testing and release automation",
        "Managed cloud infrastructure and automated scaling for services",
    ],
    "Business Analysis": [
        "Gathered business requirements and documented functional specifications",
        "Facilitated stakeholder workshops to define project scope and requirements",
        "Produced process documentation and business process diagrams",
        "Analyzed business processes to identify efficiency improvements",
        "Created user stories and acceptance criteria for development teams",
        "Conducted gap analysis between current and desired business processes",
    ],
    "Human Resources": [
        "Managed recruitment onboarding and employee relations processes",
        "Coordinated employee training programs and performance reviews",
        "Administered payroll and benefits programs for staff",
        "Handled employee grievances and maintained HR policy compliance",
        "Led talent acquisition efforts and candidate interview processes",
        "Maintained employee records and supported workforce planning",
    ],
}


def build_synthetic_dataframe() -> pd.DataFrame:
    rows = []
    for category, samples in SYNTHETIC_SAMPLES.items():
        for text in samples:
            rows.append({"Category": category, "Resume": text})
    return pd.DataFrame(rows)


class TestTrainingPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_synthetic_dataframe()
        cls.outcome = train_and_evaluate(cls.df, test_size=0.33, random_state=42)

    def test_all_nine_categories_present(self):
        self.assertEqual(len(self.outcome["labels"]), 9)
        self.assertEqual(set(self.outcome["labels"]), set(CATEGORIES))

    def test_all_three_models_trained_and_compared(self):
        self.assertEqual(
            set(self.outcome["results"].keys()),
            {"logistic_regression", "linear_svm", "multinomial_nb"},
        )
        for name, metrics in self.outcome["results"].items():
            for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
                self.assertIn(key, metrics)
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)

    def test_best_model_selected_by_macro_f1(self):
        best = self.outcome["best_model_name"]
        best_f1 = self.outcome["results"][best]["f1_macro"]
        for name, metrics in self.outcome["results"].items():
            self.assertLessEqual(metrics["f1_macro"], best_f1 + 1e-9)

    def test_confusion_matrix_shape_matches_label_count(self):
        n = len(self.outcome["labels"])
        cm = self.outcome["confusion_matrix"]
        self.assertEqual(len(cm), n)
        self.assertTrue(all(len(row) == n for row in cm))

    def test_save_artifacts_writes_expected_files(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            save_artifacts(self.outcome, model_dir=tmp_dir)
            self.assertTrue((tmp_dir / "model.pkl").exists())
            self.assertTrue((tmp_dir / "vectorizer.pkl").exists())
            self.assertTrue((tmp_dir / "metrics.json").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_vectorizer_builder_config(self):
        vec = build_vectorizer(max_features=100)
        self.assertEqual(vec.max_features, 100)
        self.assertEqual(vec.ngram_range, (1, 2))


if __name__ == "__main__":
    unittest.main()

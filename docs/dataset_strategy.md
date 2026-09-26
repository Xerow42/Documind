# Dataset Strategy

## Source

**Name:** Resume Dataset (`UpdatedResumeDataSet.csv`)
**Original author:** Gaurav Dutta Kiit, published on Kaggle (original
`gauravduttakiit/resume-dataset` URL has since gone dead)
**Working mirror (verified same 962-row, 2-column shape):**
https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset
— if this has also moved, search Kaggle for "Updated Resume Dataset"
(962 rows, `category`/`resume` columns). Also mirrored on Hugging Face
under an Apache-2.0 license.

**Size:** 962 resumes, 2 columns (`Category`, `Resume` — plain resume text, no PDFs)
**Original label set:** 25 job categories, including (with approximate counts):
Java Developer (84), Testing (70), DevOps Engineer (55), Python Developer (48),
Web Designing (45), HR (44), Hadoop (42), Sales (40), Data Science (40),
ETL Developer (40), Mechanical Engineer (40), Operations Manager (40),
Blockchain (40), Arts (36), Database (33), Health and Fitness (30), PMO (30),
Electrical Engineering (30), DotNet Developer (28), Business Analyst (28),
plus Civil Engineer, SAP Developer, Automation Testing, Network Security
Engineer, and Advocate (smaller counts).

## Why this dataset

It's real, labeled, resume-specific text (not synthetic), sized reasonably
for a TF-IDF + classical ML baseline, and — unlike a generic news/topic
dataset — several of its categories map cleanly onto genuine IT job fields,
which fits DocuMind's "resume classification for recruitment" framing.

## Honest gap vs. the requested category list

The originally requested categories were: Data Science, Software Engineering,
Web Development, Cybersecurity, Database Administration, Network
Administration, DevOps / Cloud Computing, Business Analysis, HR, Marketing.

The dataset does **not** contain a "Network Administration" or "Marketing"
category, and nothing in the other 25 labels is a reasonable proxy for
either (the closest label to networking, "Network Security Engineer", is
already used for Cybersecurity). Inventing resumes for these two categories,
or mislabeling an unrelated category as "Marketing", would mean training on
fabricated ground truth — exactly what we agreed not to do.

**Decision: ship v1 with 9 categories, dropping Network Administration and
Marketing**, documented here and in the README as a known dataset
limitation, not silently omitted.

## Category mapping (source label -> DocuMind label)

| DocuMind category | Source label(s) used | Resumes (approx.) |
|---|---|---|
| Data Science | Data Science | 40 |
| Data Engineering | Hadoop + ETL Developer | 82 |
| Software Engineering | Java Developer + Python Developer + DotNet Developer | 160 |
| Web Development | Web Designing | 45 |
| Cybersecurity | Network Security Engineer | ~30 |
| Database Administration | Database | 33 |
| DevOps / Cloud Computing | DevOps Engineer | 55 |
| Business Analysis | Business Analyst | 28 |
| Human Resources | HR | 44 |

("Data Engineering" wasn't on the original requested list, but it's a
direct, honest fit for two existing labels and strengthens the Big
Data/AI positioning of the portfolio — worth keeping unless you'd rather
drop it to stay strictly to the original 10.)

The remaining ~13 source categories (Sales, Arts, Mechanical/Electrical/
Civil Engineer, Health and Fitness, PMO, Operations Manager, SAP Developer,
Automation Testing, Blockchain, Testing, Advocate) are excluded — they
don't correspond to any target category and including them would just add
noise.

## Known limitations (to state in the README, not hide)

- **Class imbalance:** Software Engineering (160) vs. Business Analysis (28)
  is a 5.7x gap. Mitigated with `class_weight="balanced"` in training;
  evaluation reports macro-F1 (not just accuracy) so the imbalance can't
  hide poor performance on small classes.
- **Domain scope:** the model is trained on IT/business resumes scraped
  from public resume-example sites, not real hiring data — predictions on
  resumes very different in style (e.g. academic CVs) will likely be less
  reliable, and the README should say so.
- **Missing categories:** Network Administration and Marketing are not
  supported in v1 (see above). Listed explicitly as a "Future Improvements"
  item in the README rather than silently absent.
- **License:** the dataset is publicly available for research/educational
  use; the README will link back to the original Kaggle source rather than
  redistributing the raw CSV in the repo (kept out of git via
  `.gitignore`, downloaded by the user per `ml/dataset/README.md`).

# How to work in this repository

Two things trip people up in this course: **which notebook produces the file the
next one opens**, and **where your own work lives**. This page answers both.

---

## 1. Running a notebook — two routes

### Google Colab (nothing to install)
Every in-class task has a Colab twin whose first cell clones this repository into
the Colab session. Use it if your laptop is not set up, or if you are on a
borrowed machine.

| Session | Task | Open in Colab |
|---|---|---|
| 1 | Data Quality Hunt | [`session1/tasks/task_data_quality_colab.ipynb`](https://colab.research.google.com/github/dshalvardjiev/cos3121a-ml-course/blob/main/session1/tasks/task_data_quality_colab.ipynb) |

Colab keeps nothing. If you want your work, use **File → Download → .ipynb**
before you close the tab.

### Your own machine (the real setup)
```bash
git clone https://github.com/dshalvardjiev/cos3121a-ml-course.git
cd cos3121a-ml-course
pip install -r requirements.txt
pip install -e .          # optional: the notebooks find src/ on their own
jupyter lab               # or open the folder in VS Code
```
The raw datasets are committed, so there is nothing to download.

---

## 2. What produces what — the chain that matters

Each session's demo notebook **produces the file the next session's demo opens.**
That is deliberate: it is the same pipeline you would build at work, assembled a
week at a time. It also means the notebooks are not independent.

```
S1 demo_data_profiling.ipynb
      └─ writes data/processed/telco_clean.parquet
                │
S2 demo_supervised_xgboost.ipynb
      ├─ reads  data/processed/telco_clean.parquet   ← fails without the step above
      └─ writes models/churn_xgb_v1.joblib
                │
S4 demo_deploy_endpoint.ipynb
      └─ reads  models/churn_xgb_v1.joblib           ← fails without the step above
```

**`data/processed/` and `models/` are deliberately not in this repository.** You
produce them by running the notebooks, which is the point. If you clone fresh and
jump straight to Session 2's demo, it will not find its input — that is not a bug.

**What to do:** run each session's demo notebook once, in order, before the next
session. It takes a couple of minutes and it is the only prerequisite.

Files that *are* committed and always available:

| Path | Why it is here |
|---|---|
| `data/raw/*.csv` | So no in-class task ever depends on a download working |
| `session*/tasks/` | The task notebooks |
| `session*/solutions/` | Published as each session is delivered |
| `src/course_utils/` | Shared helpers imported by every notebook |
| `project/` | Phase 1 starter, report and architecture templates, dataset list |

---

## 3. Where your own work lives — your branch

From **Session 2 onwards**, each of you works on your own branch of this
repository. `main` stays as the course material; your branch is your workbook.

```bash
git checkout -b student/<yourname>      # e.g. student/maria — once, at the start
# ... work in the notebooks ...
git add -A
git commit -m "S2 task: hyperparameter sweep"
git push -u origin student/<yourname>   # first push; afterwards just: git push
```

Rules that keep this painless:

1. **Never commit to `main`.** Always check you are on your branch: `git branch --show-current`
2. **Pull before each session** so you get that week's new material:
   `git checkout main && git pull && git checkout student/<yourname> && git merge main`
3. **Do not commit data or models.** `data/processed/` and `models/` are ignored
   for everyone — they are big, they are reproducible, and they cause merge
   conflicts. Your notebooks and your written work are what belong in git.
4. **Your Phase 1 work goes in your branch**, under `project/`. Copy
   `project/phase1_starter.ipynb`, point it at your dataset, and commit as you go.

Committing regularly is also your safety net: your Phase 1 model is 35% of the
grade, and "my laptop died" is a much smaller problem when the work is pushed.

### Before any of that works
You need a GitHub account and an invitation to this repository. Tell me your
GitHub username — the placement check asks for it — and I will send the
invitation. Until then, clone and work locally; nothing is lost, you just add the
branch later.

---

## 4. Quick troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: telco_clean` | You skipped Session 1's demo | Run `session1/demo_data_profiling.ipynb` top to bottom |
| `ModuleNotFoundError: course_utils` | Notebook opened outside the repo | Open it from inside the cloned folder; the first cell walks up to find `src/` |
| `ModuleNotFoundError: xgboost` (etc.) | Dependencies not installed | `pip install -r requirements.txt` |
| Parquet errors | `pyarrow` missing | `pip install pyarrow` |
| `git push` rejected | You are on `main` | `git checkout -b student/<yourname>` and push that |

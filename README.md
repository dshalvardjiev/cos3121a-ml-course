# Machine Learning: Building Intelligent Data-Driven Applications
Course code repository — AUBG, Fall 2026. Instructor: Dimiter Shalvardjiev.

Everything used in class lives here: demo notebooks, in-class task notebooks,
solutions, shared utilities, and the project scaffolds. Exercises build on one
another across sessions:

```
S1 profile & clean data ──▶ S2 train models on it ──▶ S4 deploy S2's model ──▶ S5 audit it for bias
                             S2 segment customers     S4 call managed AI       S5 explain predictions
```

## Layout
| Path | What |
|---|---|
| `session1/ … session5/` | Demo notebooks (instructor-led) per session |
| `sessionN/tasks/` | 15-minute in-class task notebooks (students) |
| `sessionN/solutions/` | Solutions — distributed after each session |
| `src/course_utils/` | Shared helpers imported by every notebook |
| `scripts/` | Data download, defect injection, AWS access checks |
| `config/course_config.py` | Region, bucket, Bedrock model IDs — edit here, not in notebooks |
| `project/` | Phase 1 starter, report template, Phase 2 templates, dataset list |
| `data/raw` | Datasets, **committed** so nothing depends on a download |
| `data/processed`, `models/` | Produced by the notebooks, git-ignored on purpose — see WORKFLOW.md |

## Start here
New to the repository? **[WORKFLOW.md](WORKFLOW.md)** explains the two ways to run a
notebook (Colab or local), which notebook produces the file the next one opens, and
how your own branch works. Read it once and most confusion disappears.

No setup, no problem: the Session 1 task runs in Colab with nothing installed —
[open it here](https://colab.research.google.com/github/dshalvardjiev/cos3121a-ml-course/blob/main/session1/tasks/task_data_quality_colab.ipynb).

## Setup (local or SageMaker Studio)
```bash
pip install -r requirements.txt
pip install -e .                 # makes course_utils importable
python scripts/download_data.py  # fetches datasets (falls back to synthetic data offline)
python scripts/make_dirty.py     # builds the Session 1 task dataset
```
On AWS: SageMaker Studio (JupyterLab), region **us-east-1** (AWS Academy restriction).
Run `python scripts/check_aws_access.py` to see which services your account can reach.

## No AWS? No problem
Every AWS demo has a local fallback:
- All model training runs locally (scikit-learn / XGBoost).
- `USE_MOCK_AWS=1` (env var or `course_config.py`) makes Bedrock / Rekognition /
  Comprehend calls return recorded responses from `sessionN/fallback_responses/`.
- Session 4 deployment has a local FastAPI twin: `python session4/local_api.py`.

## Cost hygiene (AWS Academy credits are finite)
1. **Delete endpoints after class** — they bill while idle: `course_utils.aws.delete_endpoint(name)`.
2. Stop Studio spaces when you leave.
3. Training jobs: smallest instance that works (`ml.m5.large` is plenty here).

## Phase 1 quick start
1. Pick a dataset from `project/datasets.md` (declare by end of Session 1).
2. Copy `project/phase1_starter.ipynb`, point it at your dataset, work through it.
3. A working endpoint is expected by the start of Session 4; report (template in
   `project/`) due Session 5.

# TicketFlow

AI-powered support ticket classification. Enter a ticket subject and description and TicketFlow will:

- predict the department (Billing, Technical, HR or General),
- show the model's confidence,
- mark the ticket **Urgent** or **Normal**,
- send it to **Needs Human Review** if confidence is below 60%.

Everything is built from scratch in this project: the synthetic dataset, the trained model and the website. No external AI service, no JavaScript, no downloads of data or models.

**Stack:** Python, FastAPI, Uvicorn, Jinja2, HTML and CSS, Pandas, NumPy, scikit-learn, Joblib.

---

## Run it on Windows

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/); tick "Add python.exe to PATH" during install).

Open **PowerShell** in the project folder (the one containing `requirements.txt`) and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000> in your browser.

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"), either run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or use Command Prompt instead: `.venv\Scripts\activate.bat`.

### What happens on first start

- If `data/tickets.csv` is missing, it is generated.
- If the model files in `models/` are missing, or were saved with a different scikit-learn version, the model is trained automatically (a few seconds).
- Otherwise the saved model is loaded. **It is not retrained on every start.**

The ZIP already contains a generated dataset and a trained model, so the first start is quick.

### Optional commands

```powershell
python -m app.generate_dataset      # rebuild data/tickets.csv (same output every time)
python -m app.train_model           # retrain and print accuracy, report, confusion matrix
python -m unittest tests.test_core -v   # logic tests (no server needed)
python -m tests.smoke_test          # end-to-end test of the real web app (needs httpx)
```

### Pages

| Route | Page |
|---|---|
| `GET /` | Dashboard |
| `GET /classify` | Classification form |
| `POST /classify` | Classify the submitted ticket |
| `GET /metrics` | Model performance |
| `GET /about` | About |
| `GET /docs` | Swagger documentation (built into FastAPI) |

---

## How it works

1. **Dataset** (`app/generate_dataset.py`): 1,800 synthetic tickets (Billing 440, Technical 500, HR 440, General 420). Each department has about a dozen scenarios, each with several subjects, core sentences and optional details, filled with random amounts, dates, names and error codes. Writing style varies (formal, casual, terse, lowercase, typos, `Re:`/`Fwd:` prefixes). About 12% of tickets mention a side issue from another department, and about 4% are deliberately vague with a random label, like real messages that could go to any queue. Output is seeded and identical on each run.
2. **Preprocessing** (`app/preprocessing.py`): combine subject and body, lowercase, remove URLs, remove special characters, collapse whitespace. Training and prediction both call this same function.
3. **Training** (`app/train_model.py`): stratified 80/20 split, `TfidfVectorizer(stop_words="english", ngram_range=(1, 2))` fitted on the training split only, `LogisticRegression(C=10, max_iter=1000)`. Accuracy, precision, recall, F1, classification report and confusion matrix are computed on the test split and saved to `models/metrics.json`. The Model Performance page reads that file; nothing is typed in by hand.
4. **Prediction** (`app/model.py`): preprocess, vectorize with the saved vectorizer, `predict_proba()`. If the top probability is at least 0.60 the department is assigned, otherwise the result is **Needs Human Review**.
5. **Priority** (`app/priority.py`): rule-based. A ticket is Urgent if it contains `urgent`, `down`, `not working`, `critical` or `immediately` (whole words, case-insensitive).

### Adding urgent keywords

Edit the list at the top of `app/priority.py`:

```python
URGENT_KEYWORDS = ["urgent", "down", "not working", "critical", "immediately", "outage"]
```

### Changing the confidence threshold

Change `CONFIDENCE_THRESHOLD` in `app/train_model.py`. The pages update automatically.

---

## Project structure

```
TicketFlow/
├── app/
│   ├── main.py               FastAPI routes (thin)
│   ├── services.py           page logic, validation, session history
│   ├── model.py              loads model, predict(), auto-train if missing
│   ├── preprocessing.py      text cleaning shared by training and prediction
│   ├── priority.py           urgent keyword rules
│   ├── train_model.py        training and evaluation
│   ├── generate_dataset.py   synthetic dataset
│   ├── templates/            base, dashboard, classify, metrics, about
│   └── static/css/style.css
├── data/tickets.csv
├── models/                   ticket_classifier.joblib, tfidf_vectorizer.joblib, metrics.json
├── tests/                    test_core.py, smoke_test.py
├── requirements.txt
└── README.md
```

---

## Important notes and limitations

- **The accuracy is measured on synthetic data.** The tickets are generated from patterns written for this project, so real customer messages will usually score lower. For real use, replace `data/tickets.csv` with your own labelled tickets (same four columns) and run `python -m app.train_model`.
- **The "recent classifications" list and session counter are held in memory.** They reset when the server restarts.
- **Priority is keyword-based.** "down" also matches phrases like "shut down" and "slow down", and "isn't working" is not matched because the rule looks for the exact phrase "not working". Extend the list as needed.
- **The page loads two web fonts from Google Fonts.** With no internet connection the pages fall back to Segoe UI automatically and everything else works.
- **Form validation** returns HTTP 422 with the page re-rendered and your text preserved.

---

## What was tested

Tested in the build environment:

- Dataset generation (1,800 rows, no duplicate subject/body pairs, identical output on repeat runs).
- Model training, saved artifacts, startup from a clean state (no CSV, no model), and no retraining on the second start.
- 13 logic tests (preprocessing, priority rules, department predictions, probabilities, the 60% rule, empty-text handling).
- All four pages and the classify POST route, with real templates and real service logic: example tickets for every department, urgent and normal priority, the low-confidence review path, validation of empty/blank/too-long fields, and HTML escaping. Pages were also rendered in a real browser engine and checked visually at desktop and mobile widths.

**Not tested in the build environment:** the live FastAPI/Uvicorn server. FastAPI could not be installed there (no network access), so `app/main.py` was exercised through a stand-in that mimics FastAPI's interface rather than the real framework. `tests/smoke_test.py` is included to check the real app end to end on your machine; run it after installing the requirements. `GET /docs` is provided by FastAPI itself and was not exercised.

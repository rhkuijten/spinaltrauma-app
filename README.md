# spinaltrauma-app

Streamlit app for the spinal trauma 30-day mortality prediction model.

Split out of [spinalmetapp](https://github.com/rhkuijten/spinalmetapp) so the
spinal metastases app can keep its published URL and its original dependencies.

## Requirements

- Python 3.11 or higher (scikit-learn 1.8, needed to load `final_model.joblib`)

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```

Login and feedback are stored in a Google Sheet, so the app needs
`.streamlit/secrets.toml` with `spreadsheet_id`, `[cookie]`,
`[gcp_service_account]` and `[default_admin]`. The file is gitignored and must
not be committed.

## Deploying on Streamlit Community Cloud

- Main file: `app.py`
- Advanced settings: Python 3.12, and paste the contents of `secrets.toml`
  into the Secrets box. The Python version can only be chosen at first deploy.

`src/` and `explainability/` are vendored verbatim from the training repository.
`final_model.joblib` is pickled against those exact module paths, so do not
move or rename them.

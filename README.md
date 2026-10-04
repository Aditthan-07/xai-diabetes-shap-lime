# Explainable AI (XAI) Diabetes Decision Support

A modular Explainable AI framework for diabetes risk prediction and comparative clinical attribution using SHAP and LIME.

---

## Key Highlights

- **Leak-Free ML Pipeline:** Missing-value imputation (`SimpleImputer`) and classification (`RandomForestClassifier`) encapsulated strictly within a scikit-learn `Pipeline` fitted solely on training splits.
- **Dual-Engine XAI:** Combines game-theoretic global and local Shapley attributions (SHAP) with sparse rule-based local surrogate models (LIME).
- **Interactive Clinical Dashboard:** Real-time risk scoring, probability gauge, side-by-side attribution comparison, and 1-click borderline patient exploration via Streamlit.
- **Modular & Tested Architecture:** Decoupled modules across data ingestion, model training, and explanation services with automated `pytest` and GitHub Actions CI.

---

## Project Structure

```text
├── app.py                      # Interactive Streamlit clinical dashboard
├── data/                       # Local dataset fallback (Pima Indians Diabetes)
├── notebooks/                  # Streamlined 4-step exploratory laboratory notebook
├── src/
│   ├── data.py                 # Ingestion, cleaning, and leakage-free splitting
│   ├── model.py                # Pipeline building, training, metrics & persistence
│   └── explainers/             # SHAP, LIME, and unified PatientExplainer engines
├── tests/                      # Pytest automated test suite
├── requirements.txt            # Pinned production dependencies
└── .github/workflows/ci.yml    # Automated CI workflow
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the clinical dashboard
streamlit run app.py

# 3. Run the test suite
pytest tests/ -v
```

---

## License

Distributed under the [MIT License](https://opensource.org/licenses/MIT).

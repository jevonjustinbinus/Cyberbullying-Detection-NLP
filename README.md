# Cyberbullying Detection

Multi-tier cyberbullying detection system (NLP). Repo ini berisi kode App dan kode eksperimen. Bobot model (~755MB) di-host di Hugging Face, bukan di GitHub.

## Links
- Live App (Hugging Face Space): https://huggingface.co/spaces/jevonjustin/Cyberbullying_Detection
- Experiment Notebook (Google Colab): https://drive.google.com/drive/folders/1-X97IjBw7cwrXCVgfd1uRapNYnATdnpQ?usp=sharing
- Model files (Hugging Face): https://huggingface.co/spaces/jevonjustin/Cyberbullying_Detection/tree/main/models

## Arsitektur Model
- Tier 1  - Model individual: LR, NB, SVM, RF, KNN (TF-IDF)
- Tier 2A - Weighted Soft Voting
- Tier 2B - Hard Voting (DT + RF + XGB)
- Tier 2C - Stacking (LR + NB + SVM -> LR)
- Tier 2D - Stacking (DT + RF + XGB -> RF)
- Tier 3A - BERT + Weighted Loss
- Tier 3B - Enhanced BERT (BERT + best classical ML -> Meta-LR)

## Struktur Repo
- app.py            - Streamlit app (UI)
- predictor.py      - loading model & inference semua tier
- experiments.ipynb - kode eksperimen (training & evaluasi)
- requirements.txt  - dependencies

## Menjalankan secara lokal
File model tidak ada di repo ini. Download folder models/ dari Hugging Face Space di atas, lalu:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py
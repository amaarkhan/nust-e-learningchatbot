# ConnectEd FAQ Bot

Small RAG chatbot for the NUST ConnectEd FAQ section.

## Run

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `.env` contains `GROQ_API_KEY`.
4. Start the app:

```bash
streamlit run app.py
```

## Notes

- The bot uses a bundled FAQ dataset extracted from the ConnectEd FAQ accordion.
- The enrollment question currently points to the embedded FAQ video on the site.
- If the site FAQ changes, update `data/connected_faq.json` and restart the app.
# RadPulse

**RadPulse** is a clinical second-opinion web application designed to help junior doctors and residents interpret chest X-rays using CheXagent.

## Features
*   **Automated Report Generation:** Generates structured radiology findings and impressions from chest X-ray scans.
*   **Phrase Grounding:** Visually overlays bounding box highlights on anomalies (pneumonia, cardiomegaly, effusion, etc.).
*   **Clinical Conversational Q&A:** Allows doctors to chat with the model about specific X-ray features to double-check details.

## Tech Stack
*   **Backend:** FastAPI
*   **Frontend:** React (desktop-optimized)
*   **Model:** CheXagent / Medical VLM API

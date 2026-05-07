# DCVPA-C Analytics Dashboard

Streamlit dashboard reading registrations and visual perception scores
from your Firestore project (`dcvpa-c`).

## 1. Get a service account key

Your `firebaseConfig` (with `apiKey`) is for the web SDK — Python uses the
**Admin SDK**, which authenticates with a service account JSON.

1. Open Firebase Console → your `dcvpa-c` project.
2. ⚙ → **Project settings** → **Service accounts**.
3. Click **Generate new private key**. Confirm.
4. Save the downloaded file as `serviceAccountKey.json` next to `dashboard.py`.
5. Add it to `.gitignore`. Treat it like a password — full read/write to your DB.

## 2. Install + run locally

```bash
python -m venv .venv
source .venv/bin/activate          # on Parrot/Linux
pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at http://localhost:8501. Pick "Local file" in the sidebar and
hit connect — or use "Upload JSON" to paste the key in at runtime.

## 3. Pages

- **Overview** — totals, registrations over time, gender / residence / language breakdowns
- **Registrations** — searchable, filterable table; CSV export
- **Participant deep-dive** — pick a VPD ID + assessment date, see the full score card
  with per-domain accuracy, the Appendix-2 group totals, and grand total
- **Cohort analytics** — mean accuracy per domain across all participants, response-time
  patterns, per-level heatmap, full long-format export

## 4. Hosting options

**For private/internal use** (recommended given participant data):

- **Streamlit Community Cloud** — free for public apps, but **don't use it** here:
  participant data shouldn't sit on a public deployment.
- **Self-host on your Vasaari server** (162.214.75.152) — you already have Caddy +
  Docker. Add a service:

  ```yaml
  # docker-compose.yml fragment
  dcvpa-dashboard:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./:/app
      - ./serviceAccountKey.json:/app/serviceAccountKey.json:ro
    command: >
      bash -c "pip install -r requirements.txt &&
               streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"
    expose:
      - "8501"
  ```

  Then in Caddy:
  ```
  dashboard.dcvpa.yourdomain {
      reverse_proxy dcvpa-dashboard:8501
      basic_auth {
          admin <bcrypt-hash>
      }
  }
  ```
  Streamlit doesn't ship auth — put basic auth or your existing OIDC in front of it.

- **Cloud Run** — drop in a Dockerfile, point it at the same service account, gate
  access with IAP or Firebase Auth.

## 5. Caching

`fetch_registrations` and `fetch_all_scores_long` are cached for 5 minutes.
Use the **🔄 Refresh data** button in the sidebar to force a re-read after new
submissions.

## 6. Data shape it expects

Matches your `saveScore` and registration form code exactly:

```
registrations / {VPxxx}                      → registration document
scores / {VPxxx} / {YYYY-MM-DD} / {domain}   → { "Level N": { score, timestamp } }
```

If you change the path layout, update `fetch_scores_for_vpd` and `fetch_registrations`.

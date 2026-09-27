# Deployment guide

Backend (FastAPI) on **Render**, frontend (Vite/React) on **Vercel**. Steps marked **You** need your own
accounts or browser sign-ins.

Order: GitHub, then Earth Engine service account (optional but needed for live analysis), then Render, then
Vercel, then connect the two with CORS.

---

## 1. Push the code to GitHub

**You:** create an empty repository at https://github.com/new
* Owner: `Mrithikahub`, name: `solar-site-suitability` (or any name; update the commands below)
* Public or private, **no** README / .gitignore / license (the repo already has them)

Then, from the project folder:

```bash
git remote add origin https://github.com/Mrithikahub/solar-site-suitability.git
```

```bash
git push -u origin main
```

The first push opens a GitHub sign-in window (Git Credential Manager). Sign in there; nothing is typed
into the terminal.

---

## 2. Earth Engine service account (for live per-click analysis)

Without this the deployed API still works: every click is answered from the cached 0.5° grid and marked
"Grid estimate".

**You**, in the Google Cloud console for project `e-nebula-469422-p6`:

1. **Enable the API:** https://console.cloud.google.com/apis/library/earthengine.googleapis.com, then
   *Enable*.
2. **Create the service account:** *IAM & Admin → Service Accounts → Create service account*.
   * Name: `solarsite-api`
   * Role: **Earth Engine Resource Viewer** (`roles/earthengine.viewer`), plus
     **Service Usage Consumer** (`roles/serviceusage.serviceUsageConsumer`)
3. **Create a key:** open the service account → *Keys → Add key → Create new key → JSON*. A `.json` file
   downloads. Keep it private: it is already covered by `.gitignore`, so never move it into the repo.
4. **Register it for Earth Engine:** noncommercial projects use the project's own registration, so the
   account works once the project is registered at https://code.earthengine.google.com/register
   (you did this when you first set up Earth Engine). If requests fail with "not registered", add the
   service-account email at https://signup.earthengine.google.com/#!/service_accounts.
5. **Optional:** while the project is in Restricted Mode, complete the noncommercial eligibility check
   (Cloud console → Earth Engine → Configuration) to lift the concurrency cap.

You will paste the **entire contents** of the JSON key into Render in step 3.

---

## 3. Backend on Render

**You:** sign in at https://render.com with GitHub.

1. *New → Blueprint* and pick the repository. Render reads `render.yaml` and proposes the service
   `solarsite-api` (Python 3.11.9, free plan, Singapore region, health check `/health`).
2. Fill in the variables marked *sync: false*:

| Key | Value |
|---|---|
| `GEE_PROJECT_ID` | `e-nebula-469422-p6` |
| `GEE_SERVICE_ACCOUNT_JSON` | the whole JSON key file content, pasted as one value (optional, see step 2) |
| `ALLOWED_ORIGINS` | your Vercel URL from step 4, e.g. `https://solar-site-suitability.vercel.app` (you can come back and set this after deploying the frontend) |

3. *Apply*. The first build installs the Python stack (about 5-8 minutes).
4. Check `https://<your-service>.onrender.com/health`. `live_gee.enabled` should be `true` and
   `last_error` `null` after the first live click.

The free plan sleeps after 15 minutes without traffic, so the first request after a pause takes about
a minute. The model and grid files are committed, so no build-time data download is needed.

---

## 4. Frontend on Vercel

**You:** sign in at https://vercel.com with GitHub.

1. *Add New → Project* and import the repository.
2. **Root Directory:** `frontend`. The framework preset (Vite), build command and output directory come
   from `frontend/vercel.json`.
3. **Environment variable:** `VITE_API_URL` = `https://<your-service>.onrender.com` (no trailing slash).
4. *Deploy*. Deep links such as `/map?lat=9.33&lon=78.39` work through the SPA rewrite in `vercel.json`.

---

## 5. Connect them (CORS)

Back in Render, set `ALLOWED_ORIGINS` to the Vercel production URL (comma-separate several).
`ALLOWED_ORIGIN_REGEX` already allows `*.vercel.app` preview deployments. Saving the variable redeploys
the API.

Smoke test in the deployed site:

* **Map:** click Bhadla in "Try a famous solar park". It should score High, with "Live Earth Engine" (or
  "Grid estimate" without a service account).
* **Download report:** a PDF should download.
* **Methodology page:** figures and tables should load. They are static files shipped with the frontend.

---

## Updating

Push to `main`. Render and Vercel both redeploy automatically. After retraining models locally, commit the
updated `model/*.pkl`, `data/processed/*` and, if figures changed, run
`python scripts/sync_frontend_assets.py` before committing.

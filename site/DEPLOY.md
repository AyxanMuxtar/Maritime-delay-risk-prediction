# Deploying to Vercel for free

This guide takes you from "site files in a folder" to "live URL anyone can visit" in about 10 minutes. Total cost: **$0/month, forever**.

## How the architecture works

```
GitHub Actions (already set up)        Vercel (deploys automatically)
─────────────────────────────          ────────────────────────────────
1st of each month, 06:00 UTC           Watches your repo for git pushes
  ↓
runs pipeline.py                       When it sees the new CSVs,
  ↓                                    rebuilds the static site
generates predictions/YYYY-MM/             ↓
{daily,monthly}.csv                    Serves caspian-risk.vercel.app
  ↓
commits CSVs back to repo
  ↓
─── triggers Vercel ─→
```

You don't need to deploy manually. Once connected, every git push triggers a redeploy. The monthly cron commit is just another git push.

## Why this architecture (and the trade-off)

Vercel's free Hobby tier doesn't run Python — it's for static files and Node/Edge functions only. So the actual ML pipeline can't run on Vercel. But that's fine: GitHub Actions already runs your pipeline for free, and Vercel only needs to serve the prediction CSVs as a static site. Total cost: $0 across both.

The trade-off: if a user requests a forecast at any time other than the 1st of the month, they see the most recent forecast (which could be up to 30 days old). For a maritime delay-risk site updated monthly, that's expected behavior.

## Prerequisites

1. The project committed to a GitHub repo. If it isn't yet, do this first:
   ```bash
   cd C:\Users\user\project
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/caspian-risk.git
   git branch -M main
   git push -u origin main
   ```

2. The site files in a `site/` folder at the repo root. If you used the deliverable from Day 9, this is already done.

3. The predictions folder at the repo root, populated by the pipeline. The site fetches `predictions/YYYY-MM/daily.csv` and `monthly.csv` relative to itself.

## Important — file layout

Vercel needs the site files and the predictions folder accessible to each other. Two options:

### Option A — site files at repo root (simplest)

Move `index.html`, `style.css`, `script.js`, etc. directly into the repo root. The `predictions/` folder is also at the root. Now `fetch('predictions/2026-05/daily.csv')` works from `index.html`.

Repo layout:
```
caspian-risk/
├── index.html
├── archive.html
├── about.html
├── style.css
├── script.js
├── vercel.json
├── predictions/
│   └── 2026-05/
│       ├── daily.csv
│       └── monthly.csv
├── src/
├── notebooks/
└── ... rest of project
```

### Option B — site files in /site/ subfolder

Keep the project clean by putting site files in a `site/` subdirectory and copying predictions in via the GitHub Action. Add this step to `.github/workflows/monthly-pipeline.yml` after the predictions are generated:

```yaml
- name: Copy predictions into site folder
  run: |
    mkdir -p site/predictions
    cp -r predictions/* site/predictions/
```

Then in Vercel's project settings, set the **Root Directory** to `site/`.

I recommend **Option A** for a student project. Less moving parts.

## Step-by-step deployment

### 1. Push to GitHub

If you haven't already:
```bash
git add .
git commit -m "Add deployment site"
git push
```

### 2. Sign up for Vercel

Go to [vercel.com/signup](https://vercel.com/signup). Sign up with **"Continue with GitHub"** — it's the easiest way because Vercel can then read your repos directly. No credit card needed.

### 3. Import the project

After signup, you'll land on the dashboard. Click **"Add New..." → "Project"**.

Vercel shows a list of your GitHub repos. Find `caspian-risk` (or whatever you named it). Click **"Import"**.

### 4. Configure the build

Vercel will try to auto-detect a framework. Tell it there isn't one:

- **Framework Preset**: `Other`
- **Root Directory**: `./` (default, if using Option A) or `site/` (if Option B)
- **Build Command**: leave empty
- **Output Directory**: leave empty (it'll serve from the root directory)
- **Install Command**: leave empty

Click **"Deploy"**.

### 5. Wait ~30 seconds

Vercel uploads, processes, and serves the static files. You'll see a confetti animation when it's done, plus a URL like:

```
https://caspian-risk-xxxxxxxx.vercel.app
```

That's your live site. Click it — the page should load with the May 2026 sample forecast.

### 6. (Optional) Custom domain

If you have a domain like `caspian-risk.example.com`:

1. In your Vercel project, go to **Settings → Domains**
2. Add your domain
3. Vercel gives you DNS records to configure at your domain registrar
4. Once DNS propagates (1 minute to a few hours), the site serves from your domain over HTTPS

Free domains: try [is-a.dev](https://is-a.dev) or [js.org](https://js.org) — both offer free dev-focused subdomains for open-source projects.

### 7. Set up auto-deploy on every commit (already done)

This is automatic with the GitHub integration. Every time you `git push`, Vercel rebuilds and redeploys within ~30 seconds.

For the monthly cron, this means:
- 1st of month, 06:00 UTC: GitHub Actions runs `pipeline.py`
- Action commits `predictions/YYYY-MM/*.csv` back to the repo
- Vercel sees the push, rebuilds the static site
- ~30 seconds later, the new month's forecast is live at your URL

Zero manual intervention.

## Verifying it works

Visit your `*.vercel.app` URL. You should see:

- The hero section showing **"Forecast · May 2026"** (from the sample data)
- Four stat cards (high-risk days, high-risk months, etc.)
- Five city cards (Baku, Aktau, Anzali, Turkmenbashi, Makhachkala) each with a 31-day calendar grid
- Color-coded cells: green (low risk), amber (elevated), red (high), gray dashed (climatology)
- Hover over any cell to see exact probability and source
- Footer with Open-Meteo attribution

If the page says "No forecast data found", the predictions folder isn't where the site expects it. Check that `predictions/2026-05/daily.csv` is at the same path level as `index.html` in your deployment.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "No forecast data found" | predictions/ not at expected path | Confirm `predictions/YYYY-MM/daily.csv` exists in the repo at the same level as `index.html` |
| Page renders but city cards are empty | CSV column names don't match | Compare your `daily.csv` headers against `city,date,day_of_month,probability,prediction,source` |
| 404 on a CSV | The month folder doesn't exist yet | Wait until next monthly cron, or run `pipeline.py` manually to generate one |
| Vercel build fails | Build command misconfigured | Set Build Command and Output Directory to empty in project settings |
| Custom domain not connecting | DNS not propagated | Wait up to 24 hours; verify CNAME / A records at registrar |

## Limits to know about

You won't hit them with a student project, but for awareness:

- **100 GB bandwidth/month** (about 100,000 monthly visitors)
- **Personal/non-commercial use only** on the Hobby tier — adding ads or selling subscriptions violates the terms
- **100 deployments/day** (you'd have to push commits in a tight loop to hit this)
- **Build time capped at 45 minutes** (irrelevant for a static site that takes ~30s)

If you ever blow through these, Vercel pauses the project for 30 days, sends an email, and offers Pro at $20/month. For a student sprint that won't happen.

## What if I want even simpler?

Two alternatives if Vercel feels heavy:

- **GitHub Pages** — same shape, possibly simpler. Push site files to a `gh-pages` branch (or use the `/docs` folder on `main`). URL is `username.github.io/caspian-risk`. No account beyond GitHub. Same $0 cost, similar performance.

- **Netlify** — basically a clone of Vercel for our purposes. Drag-and-drop deploy or GitHub connect. Same $0.

I picked Vercel because the GitHub integration is the smoothest of the three. But all three would work for this project.

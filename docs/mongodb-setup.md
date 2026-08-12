# MongoDB Atlas â€” free cluster setup

Fifteen minutes. The M0 tier is free permanently (512 MB), not a trial.

Everything in this project degrades gracefully without Mongo â€” the pipeline runs
on SQLite alone. But the raw-document store is what makes "genuine NoSQL usage"
a fact rather than a claim, so it is worth doing once, properly.

---

## 1. Create the cluster

1. <https://www.mongodb.com/cloud/atlas/register> â€” sign up
2. **Create a deployment** â†’ choose **M0 Free**
3. Provider/region: pick the one nearest you (`ap-south-1` Mumbai, if in India)
4. Name it `pulseiq` and create

Wait 1â€“3 minutes for provisioning.

## 2. Database user

**Security â†’ Database Access â†’ Add New Database User**

- Authentication: Password
- Username: `pulseiq_app`
- Password: use **Autogenerate Secure Password** and copy it
- Role: **Read and write to any database**

Do not reuse a password from anywhere else, and do not put it in any file other
than `.env`.

## 3. Network access

**Security â†’ Network Access â†’ Add IP Address**

- **Add Current IP Address** for local development.
- If your ISP rotates your IP, you will need to re-add it. `0.0.0.0/0` (allow
  from anywhere) works but is genuinely bad practice â€” on a free tier holding
  non-sensitive public data it is a defensible shortcut, but say so out loud
  rather than doing it silently. Prefer adding your IP.

## 4. Connection string

**Deployment â†’ Database â†’ Connect â†’ Drivers â†’ Python**

Copy the string. It looks like:

```
mongodb+srv://pulseiq_app:<password>@example.invalid/?retryWrites=true&w=majority
```

Replace `<password>` with the generated password. If it contains `@`, `:`, `/`
or `#`, URL-encode those characters or the URI will not parse.

## 5. Configure and verify

Add to `.env` (never `.env.example`, never a commit):

```
MONGODB_URI=mongodb+srv://pulseiq_app:YOUR_PASSWORD@example.invalid/?retryWrites=true&w=majority  # pragma: allowlist secret
MONGODB_DB=pulseiq
MONGODB_RAW_COLLECTION=raw_scrapes
```

Then:

```powershell
python -m pulseiq.storage.healthcheck
```

Expected:

```
component     status  detail
relational    OK      sqlite:///./pulseiq.db | price_snapshots=0, reviews=0
mongodb       OK      mongodb+srv://***:***@pulseiq.xxxxx... | db=pulseiq ... docs~=0
mlflow        OK      sqlite:///mlflow.db | 1 experiment(s)
```

The health check masks credentials, so its output is safe to paste or screenshot.

### If mongodb reports FAIL

| Symptom | Cause |
|---|---|
| `ServerSelectionTimeoutError` | IP not on the allowlist (step 3), or a firewall blocking outbound 27017 |
| `AuthenticationFailed` | Wrong password, or unencoded special characters in the URI |
| `InvalidURI` | The `<password>` placeholder was never replaced |
| Hangs then times out | `mongodb://` instead of `mongodb+srv://` for an Atlas cluster |

## 6. Ingest

```powershell
python -m pulseiq.ingestion.run_ingest --open-prices --max-series 200
```

Writes raw documents to Mongo in batches and cleaned rows to SQL. Re-running is
idempotent on the SQL side â€” a `UNIQUE(product_name, observed_on)` constraint
means a second run reports `skipped_existing` rather than duplicating.

Verify in the Atlas UI: **Browse Collections â†’ pulseiq â†’ raw_scrapes**. Each
document carries `source`, `run_id`, `ingested_at` and the raw `payload`.

## 7. Watch the 512 MB limit

Each raw document is roughly 300â€“500 bytes, so 512 MB is ~1 M documents â€” ample.
But if you later schedule a daily scraper, add a TTL index so old raw documents
expire rather than accumulating:

```javascript
db.raw_scrapes.createIndex({ "ingested_at": 1 }, { expireAfterSeconds: 7776000 })
```

That drops documents after 90 days. The cleaned rows in SQL are the permanent
record; Mongo is the audit trail.

---

## Why two stores at all

The question an interviewer will ask, and the answer that is actually true:

Raw scrape payloads are schemaless and nested, and their shape changes whenever a
site changes its markup. Forcing them into columns discards information needed to
debug a failed run. Cleaned rows need constraints, indexes and joins, which
document stores handle poorly. So the raw document is the audit trail in Mongo,
and the typed row is the queryable record in SQL.

The `run_id` on every document is what makes "show me everything from the run
that broke" a real query rather than a grep.

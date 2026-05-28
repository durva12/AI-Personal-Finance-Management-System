# FinanceAI — Complete Setup Guide

## Project Structure
```
financeai/
├── app.py                    # Main Flask application
├── database.py               # SQLAlchemy models (User, FinancialGoal, FinanceAnalysis)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment config template
├── models/
│   ├── __init__.py
│   ├── kmeans_clustering.py  # Fixed K-Means (8 clusters, weighted distance)
│   └── rl_finance_agent.py   # Fixed DQN agent + health score
└── templates/
    ├── index.html            # Landing page
    ├── login.html            # Login + Signup
    ├── admin_login.html      # Admin login
    ├── admin_dashboard.html  # Admin panel
    └── dashboard.html        # Main app dashboard
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- PostgreSQL (running locally or remote)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — update DATABASE_URL with your PostgreSQL credentials
```

### 4. Create PostgreSQL database
```sql
CREATE DATABASE financeai;
```

### 5. Run the app
```bash
python app.py
```

The app:
- Auto-creates all tables on first run (`db.create_all()`)
- Auto-creates the admin user if not present
- Pre-trains K-Means and DQN on startup (~5–10 seconds)

Open: http://localhost:5000

---

## Database Migration (if you change models later)

```bash
flask db init          # First time only
flask db migrate -m "your change description"
flask db upgrade
```

---

## Test Credentials

| Role  | Username / Email              | Password   |
|-------|-------------------------------|------------|
| Admin | admin (→ admin@financeai.admin) | admin123 |
| User  | Sign up at /signup            | your choice |

---

## .env Example
```
SECRET_KEY=change-this-to-a-long-random-string-in-production
FLASK_ENV=development
DATABASE_URL=postgresql://postgres:password@localhost:5432/financeai
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

---

## Bug Fixes Summary

### 1. K-Means always returning "Lifestyle Spender" — FIXED
- **Root cause**: Only 5 clusters with insufficient variance in synthetic training data; all input vectors fell near the "Lifestyle Spender" centroid
- **Fix**: Expanded to 8 well-separated clusters, applied weighted Euclidean distance (savings + debt ratios get 2× weight as most discriminating features), increased synthetic data to 4000 samples with realistic per-cluster variance

### 2. DQN giving "Excellent" score on low income — FIXED
- **Root cause**: Reward function only measured score *improvement*, not absolute financial health. Starting from a low baseline still yielded positive reward.
- **Fix**: Rebuilt health score with 6 independent components (savings ratio, expense-to-income, housing ratio, debt burden, discretionary control, healthcare). Added hard cap: income < ₹15,000 + expenses > 85% of income caps score at 35 ("Critical/Needs Work"). Low income + high expenses now always produces a low score.

### 3. Skip goal still shows target — FIXED
- **Root cause**: The skip button called `closeGoalModal()` without persisting the skip intent. On reload, `goal = None` so modal re-appeared. `/api/goal/current` returned the first non-null goal regardless.
- **Fix**: Skip now calls `/api/goal` with `{skipped: true}`. A `FinancialGoal` record with `skipped=True` is saved to DB. All goal queries check `skipped=False`. Dashboard only shows modal when *no goal record at all* exists (`show_goal_modal = goal is None`).

### 4. Double `<<nav` tag in dashboard.html — FIXED
- Simple HTML typo causing broken page layout

### 5. City cost-of-living support — ADDED
- City selection on signup and dashboard
- CITY_MULTIPLIERS dict adjusts housing/transport thresholds (Mumbai 1.4×, Bengaluru 1.2×, etc.)
- Mumbai user with 41% income on rent is not unfairly penalised

### 6. Fixed expense toggle — IMPROVED
- Visual feedback when marking a category as fixed (highlighted card)
- Fixed categories skip AI recommendations and show "🔒 Fixed" pill in budget table

### 7. Admin `avg_score` stat — ADDED
- Admin dashboard now shows average health score across all analyses

### 8. Cascade deletes — FIXED
- Deleting a user now correctly deletes their analyses and goals (cascade)

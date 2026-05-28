# app.py — FinanceAI Production Application
import os, json
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, session)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from flask_migrate import Migrate
from dotenv import load_dotenv

from database import db, User, FinancialGoal, FinanceAnalysis
from models.rl_finance_agent import (
    DQNFinanceAgent, FinanceEnvironment, CATEGORIES,
    compute_health_score, score_label, get_pretrained_agent, CITY_MULTIPLIERS
)
from models.kmeans_clustering import classify_user, get_kmeans_model, CLUSTER_PROFILES
from models.recommendation_engine import generate_recommendations
from models.pdf_report import generate_pdf_report

load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-in-prod')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 'postgresql://postgres:password@localhost:5432/financeai'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# ─── Pre-train models at startup ──────────────────────────────────────────────

print("⚙️  Training K-Means Spending Classifier (8 clusters)…")
_km = get_kmeans_model()
print(f"✅ K-Means ready — {len(_km.inertia_history)} iterations")

print("⚙️  Pre-training DQN Finance Agent…")
agent = get_pretrained_agent(income=50000, episodes=300)
print(f"✅ DQN ready — ε={agent.epsilon:.4f}")

# ─── Create admin on first run ────────────────────────────────────────────────

with app.app_context():
    db.create_all()

    admin_email = 'durvaajoshi@gmail.com'
    admin_pass = 'root123'

    if not User.query.filter_by(email=admin_email).first():
        admin = User(
            name='Admin',
            email=admin_email,
            is_admin=True
        )

        admin.set_password(admin_pass)

        db.session.add(admin)
        db.session.commit()

        print("✅ Admin account created")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _simulate_history(income, expenses, city='Other', months=12):
    """Run DQN agent forward to build 12-month projection."""
    env = FinanceEnvironment(income=income, city=city)
    for cat in CATEGORIES:
        env.budget[cat] = expenses.get(cat, income / len(CATEGORIES))
    history = []
    state = env._get_state()
    for m in range(1, months + 1):
        action = agent.act(state, training=False)
        state, reward, done, info = env.step(action)
        history.append({
            "month": m,
            "score": round(float(info["score"]), 1),
            "savings": round(float(env.budget["Savings"]), 2),
            "debt": round(float(env.debt), 2),
            "reward": round(float(reward), 4),
        })
        if done:
            break
    return history


def _goal_suggestions(goal, income, expenses, rec, fixed_categories=None):
    """Build goal-progress info. Returns None if goal is skipped."""
    if goal is None or goal.skipped:
        return None
    fixed = set(fixed_categories or [])
    monthly_target  = float(goal.monthly_target)
    current_savings = float(expenses.get('Savings', 0))
    on_track        = current_savings >= monthly_target
    gap             = monthly_target - current_savings
    tips = []
    if not on_track:
        for cat, info in rec.get('budget', {}).items():
            if cat in fixed or cat == 'Savings':
                continue
            diff = float(info.get('current', 0)) - float(info.get('recommended', 0))
            if diff > 0:
                tips.append({
                    "action": "reduce", "category": cat,
                    "amount": round(diff, 0),
                    "message": f"Cut {cat} by ₹{diff:,.0f} to free up cash"
                })
        tips.sort(key=lambda x: x['amount'], reverse=True)
        tips = tips[:3]
    projected = current_savings * goal.duration_months
    progress  = min(100, round((projected / goal.target_amount) * 100, 1)) if goal.target_amount else 0
    return {
        "goal_label":       goal.goal_label,
        "target_amount":    float(goal.target_amount),
        "monthly_target":   monthly_target,
        "current_savings":  current_savings,
        "on_track":         on_track,
        "gap":              round(float(gap), 2),
        "progress_pct":     progress,
        "months_remaining": goal.duration_months,
        "gap_message": (
            f"You need ₹{gap:,.0f} more/month to stay on track."
            if not on_track else
            f"Great! You are ₹{current_savings - monthly_target:,.0f} ahead of your monthly target."
        ),
        "tips": tips,
    }


# ─── Public Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated and not current_user.is_admin:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()
        if user and user.check_password(password) and not user.is_admin:
            login_user(user, remember=request.form.get('remember') == 'on')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html', page='login')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        email   = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        city     = request.form.get('city', 'Other')
        if not name or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('login.html', page='signup')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('login.html', page='signup')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('login.html', page='signup')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('login.html', page='signup')
        user = User(name=name, email=email, city=city)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash('Account created! Set your first financial goal below.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('login.html', page='signup',
                           cities=list(CITY_MULTIPLIERS.keys()))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# ─── Admin ────────────────────────────────────────────────────────────────────

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated and current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        admin_email = username + '@financeai.admin'
        user = User.query.filter_by(email=admin_email, is_admin=True).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Admin access required.', 'danger')
        return redirect(url_for('index'))
    users    = User.query.filter_by(is_admin=False).order_by(User.created_at.desc()).all()
    analyses = FinanceAnalysis.query.order_by(FinanceAnalysis.created_at.desc()).limit(20).all()
    goals    = FinancialGoal.query.filter_by(skipped=False)\
                            .order_by(FinancialGoal.created_at.desc()).limit(20).all()
    stats = {
        'total_users':    User.query.filter_by(is_admin=False).count(),
        'total_analyses': FinanceAnalysis.query.count(),
        'total_goals':    FinancialGoal.query.filter_by(skipped=False).count(),
        'avg_score': round(
            db.session.query(db.func.avg(FinanceAnalysis.health_score)).scalar() or 0, 1
        ),
    }
    return render_template('admin_dashboard.html',
                           users=users, analyses=analyses, goals=goals, stats=stats)


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_dashboard'))
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.email} deleted.', 'success')
    return redirect(url_for('admin_dashboard'))


# ─── Goal API ─────────────────────────────────────────────────────────────────

@app.route('/api/goal', methods=['POST'])
@login_required
def save_goal():
    data     = request.get_json(force=True)
    skipped  = data.get('skipped', False)

    # Deactivate any existing active goals
    FinancialGoal.query.filter_by(user_id=current_user.id, is_active=True)\
                       .update({'is_active': False})

    if skipped:
        goal = FinancialGoal(
            user_id=current_user.id,
            goal_type='skipped', goal_label='Skipped',
            target_amount=0, duration_months=1, monthly_target=0,
            is_active=True, skipped=True, status='active'
        )
        db.session.add(goal)
        db.session.commit()
        return jsonify({'status': 'skipped'})

    target   = float(data.get('target_amount', 0))
    duration = int(data.get('duration_months', 12))
    if target <= 0 or duration <= 0:
        return jsonify({'error': 'Invalid goal parameters'}), 400

    goal = FinancialGoal(
        user_id=current_user.id,
        goal_type=data.get('goal_type', 'custom'),
        goal_label=data.get('goal_label', 'My Financial Goal'),
        target_amount=target, duration_months=duration,
        monthly_target=round(target / duration, 2),
        is_active=True, skipped=False, status='active'
    )
    db.session.add(goal)
    db.session.commit()
    return jsonify({
        'status': 'saved', 'goal_id': goal.id,
        'monthly_target': goal.monthly_target,
        'goal_label': goal.goal_label
    })


@app.route('/api/goal/current', methods=['GET'])
@login_required
def get_current_goal():
    goal = FinancialGoal.query\
           .filter_by(user_id=current_user.id, is_active=True)\
           .order_by(FinancialGoal.created_at.desc()).first()
    if not goal or goal.skipped:
        return jsonify({'goal': None})
    return jsonify({'goal': {
        'id': goal.id, 'goal_type': goal.goal_type, 'goal_label': goal.goal_label,
        'target_amount': goal.target_amount,
        'duration_months': goal.duration_months,
        'monthly_target': goal.monthly_target,
        'status': goal.status,
    }})


@app.route('/api/goal/edit/<int:goal_id>', methods=['POST'])
@login_required
def edit_goal(goal_id):
    """Edit an existing active goal — ownership enforced."""
    goal = db.session.get(FinancialGoal, goal_id)
    if not goal:
        return jsonify({'error': 'Goal not found'}), 404
    if goal.user_id != current_user.id:
        return jsonify({'error': 'Unauthorised'}), 403
    if goal.skipped or goal.status == 'cancelled':
        return jsonify({'error': 'Cannot edit this goal'}), 400

    data  = request.get_json(force=True)
    label = data.get('goal_label', '').strip()
    try:
        target   = float(data.get('target_amount', 0))
        duration = int(data.get('duration_months', 12))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid numbers'}), 400

    if not label:
        return jsonify({'error': 'Goal name cannot be empty'}), 400
    if target <= 0:
        return jsonify({'error': 'Target amount must be greater than zero'}), 400
    if duration < 1 or duration > 120:
        return jsonify({'error': 'Duration must be between 1 and 120 months'}), 400

    goal.goal_label      = label
    goal.goal_type       = data.get('goal_type', goal.goal_type)
    goal.target_amount   = target
    goal.duration_months = duration
    goal.monthly_target  = round(target / duration, 2)
    db.session.commit()

    return jsonify({
        'status':         'updated',
        'goal_id':        goal.id,
        'goal_label':     goal.goal_label,
        'target_amount':  goal.target_amount,
        'duration_months': goal.duration_months,
        'monthly_target': goal.monthly_target,
    })


@app.route('/api/goal/cancel/<int:goal_id>', methods=['POST'])
@login_required
def cancel_goal(goal_id):
    """Soft-cancel a goal — keeps the DB row for audit history."""
    goal = db.session.get(FinancialGoal, goal_id)
    if not goal:
        return jsonify({'error': 'Goal not found'}), 404
    if goal.user_id != current_user.id:
        return jsonify({'error': 'Unauthorised'}), 403
    if goal.status == 'cancelled':
        return jsonify({'error': 'Goal is already cancelled'}), 400

    goal.soft_cancel()
    db.session.commit()
    return jsonify({'status': 'cancelled', 'goal_id': goal.id})


# ─── Dashboard ────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    goal = FinancialGoal.query\
           .filter_by(user_id=current_user.id, is_active=True)\
           .order_by(FinancialGoal.created_at.desc()).first()
    # Only show non-skipped, non-cancelled goals
    active_goal = None
    if goal and not goal.skipped and goal.status != 'cancelled':
        active_goal = goal
    show_goal_modal = (goal is None)
    return render_template('dashboard.html',
                           goal=active_goal,
                           show_goal_modal=show_goal_modal,
                           user=current_user,
                           cities=list(CITY_MULTIPLIERS.keys()))


# ─── Finance Analysis API ─────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        data             = request.get_json(force=True)
        income           = float(data.get('income', 50000))
        city             = data.get('city', current_user.city or 'Other')
        raw              = data.get('expenses', {})
        expenses         = {cat: float(raw.get(cat, 0)) for cat in CATEGORIES}
        fixed_categories = list(data.get('fixed_categories', []) or [])

        # Fetch last 12 months of history for trend analysis
        past = FinanceAnalysis.query\
            .filter_by(user_id=current_user.id)\
            .order_by(FinanceAnalysis.created_at.desc())\
            .limit(12).all()
        history_dicts = [{"expenses": a.expenses, "income": a.income} for a in past]

        # K-Means cluster (run first so label feeds into recommendation engine)
        cluster_result = classify_user(income, expenses)

        # Dynamic recommendation engine (replaces static agent.recommend_budget)
        rec = generate_recommendations(
            income=income,
            expenses=expenses,
            city=city,
            fixed_categories=fixed_categories,
            history=history_dicts,
            cluster_label=cluster_result['label'],
        )

        # 12-month DQN projection
        timeline = _simulate_history(income, expenses, city=city, months=12)

        # Goal progress (only if not skipped)
        active_goal = FinancialGoal.query\
            .filter_by(user_id=current_user.id, is_active=True)\
            .order_by(FinancialGoal.created_at.desc()).first()
        goal_analysis = _goal_suggestions(active_goal, income, expenses, rec, fixed_categories)

        # Save analysis to DB
        record = FinanceAnalysis(
            user_id=current_user.id,
            goal_id=active_goal.id if (active_goal and not active_goal.skipped) else None,
            income=income, expenses=expenses,
            health_score=rec['health_score'],
            cluster_label=cluster_result['label'],
            recommendations=rec,
            goal_progress=goal_analysis['progress_pct'] if goal_analysis else None
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            "status":           "success",
            "income":           income,
            "city":             city,
            "recommendations":  rec,        # now contains .recommendations[] and .healthy[]
            "timeline":         timeline,
            "cluster":          cluster_result,
            "goal_analysis":    goal_analysis,
            "fixed_categories": fixed_categories,
            "has_history":      len(history_dicts) > 0,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/demo', methods=['GET'])
def demo():
    demo_income   = 65000
    # Intentionally unbalanced so the engine has something interesting to say
    demo_expenses = {
        "Housing": 18000, "Food": 12000, "Transport": 5000,
        "Entertainment": 9500, "Healthcare": 2500, "Savings": 4000,
        "Debt Repayment": 7000, "Miscellaneous": 6000
    }
    cluster_result = classify_user(demo_income, demo_expenses)
    rec    = generate_recommendations(demo_income, demo_expenses, city='Pune',
                                       cluster_label=cluster_result['label'])
    timeline = _simulate_history(demo_income, demo_expenses, city='Pune', months=12)
    return jsonify({
        "status": "success", "income": demo_income, "city": "Pune",
        "recommendations": rec, "timeline": timeline, "cluster": cluster_result,
        "goal_analysis": None, "fixed_categories": [], "is_demo": True,
        "has_history": False,
    })


@app.route('/api/cluster_profiles', methods=['GET'])
def cluster_profiles():
    return jsonify({"profiles": CLUSTER_PROFILES})


@app.route('/api/user/history', methods=['GET'])
@login_required
def user_history():
    analyses = FinanceAnalysis.query\
        .filter_by(user_id=current_user.id)\
        .order_by(FinanceAnalysis.created_at.desc()).limit(10).all()
    return jsonify({"history": [
        {"id": a.id, "income": a.income, "health_score": a.health_score,
         "cluster_label": a.cluster_label, "goal_progress": a.goal_progress,
         "created_at": a.created_at.isoformat()}
        for a in analyses
    ]})


@app.route('/api/retrain', methods=['POST'])
@login_required
def retrain():
    global agent
    data     = request.get_json(force=True)
    episodes = int(data.get('episodes', 200))
    income   = float(data.get('income', 50000))
    env      = FinanceEnvironment(income=income)
    agent.train(env, episodes=episodes, verbose=False)
    return jsonify({
        "status": "retrained",
        "total_episodes": len(agent.training_history),
        "epsilon": round(agent.epsilon, 4),
        "last_score": agent.training_history[-1]['score'] if agent.training_history else 0
    })


@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/download_report', methods=['POST'])
@login_required
def download_report():
    """Generate and stream a PDF report from the latest analysis data."""
    try:
        from flask import send_file
        data          = request.get_json(force=True)
        income        = float(data.get('income', 0))
        city          = data.get('city', current_user.city or 'Other')
        rec           = data.get('recommendations', {})
        cluster       = data.get('cluster', {})
        timeline      = data.get('timeline', [])
        goal_analysis = data.get('goal_analysis', None)

        pdf_bytes = generate_pdf_report(
            user_name=current_user.name,
            income=income,
            city=city,
            rec=rec,
            cluster=cluster,
            timeline=timeline,
            goal_analysis=goal_analysis,
        )

        from io import BytesIO
        buf = BytesIO(pdf_bytes)
        buf.seek(0)
        filename = f"FinanceAI_Report_{current_user.name.replace(' ','_')}.pdf"
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True, download_name=filename)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

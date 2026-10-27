import os
from collections import defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, render_template_string, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from jinja2 import ChoiceLoader, FileSystemLoader
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from sklearn.linear_model import LinearRegression

load_dotenv()

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///digital_twin.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure Jinja can resolve both templates/ and templates/user/
templates_path = os.path.join(app.root_path, 'templates')
user_templates_path = os.path.join(templates_path, 'user')
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(templates_path),
    FileSystemLoader(user_templates_path),
])

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ------------------------- Context Processor -------------------------
@app.context_processor
def utility_processor():
    return dict(datetime=datetime)

# ------------------------- Database Models -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, default=25)
    occupation = db.Column(db.String(100), default='Student')
    monthly_income = db.Column(db.Float, default=50000.0)
    monthly_expenses = db.Column(db.Float, default=35000.0)
    savings = db.Column(db.Float, default=100000.0)
    study_hours_per_week = db.Column(db.Float, default=20.0)
    fitness_hours_per_week = db.Column(db.Float, default=5.0)
    sleep_hours_per_day = db.Column(db.Float, default=7.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    transactions = db.relationship('Transaction', backref='user', lazy=True)
    study_logs = db.relationship('StudyLog', backref='user', lazy=True)
    fitness_logs = db.relationship('FitnessLog', backref='user', lazy=True)
    goals = db.relationship('Goal', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    category = db.Column(db.String(50))
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10))  # 'income' or 'expense'


class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    hours = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100))
    productivity_score = db.Column(db.Integer)


class FitnessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    activity = db.Column(db.String(50))
    duration_min = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer)


class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    target_date = db.Column(db.Date)
    achieved = db.Column(db.Boolean, default=False)


# ------------------------- Helper Functions -------------------------
def generate_forecast(data, periods=6):
    if len(data) < 2:
        return [data[-1] if data else 0] * periods
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array(range(len(data), len(data) + periods)).reshape(-1, 1)
    forecast = model.predict(future_X)
    noise = np.random.normal(0, np.std(y) * 0.1 if np.std(y) > 0 else 1, periods)
    return (forecast + noise).tolist()


def calculate_financial_metrics(user):
    income = sum(t.amount for t in user.transactions if t.type == 'income')
    expenses = sum(t.amount for t in user.transactions if t.type == 'expense')
    net = income - expenses
    rate = (net / income * 100) if income > 0 else 0
    return {'income': income, 'expenses': expenses, 'net': net, 'rate': rate}


def predict_gpa(user):
    logs = user.study_logs
    if not logs:
        return {'gpa': 3.0, 'efficiency': 70, 'recommendation': 'Log study sessions to get predictions.'}
    total_hours = sum(l.hours for l in logs)
    avg_prod = sum(l.productivity_score for l in logs) / len(logs)
    gpa = 2.0 + (total_hours / 100) + (avg_prod / 100)
    gpa = min(4.0, max(1.0, gpa))
    if total_hours < 20:
        rec = "Increase study hours."
    elif avg_prod < 70:
        rec = "Improve focus and quality."
    else:
        rec = "Keep up the good work!"
    return {'gpa': round(gpa, 2), 'efficiency': round(avg_prod, 1), 'recommendation': rec}


# ------------------------- Default Admin Creator -------------------------
def create_default_admin():
    """Create a default admin user if none exists."""
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@digitaltwin.ai'
        )
        admin.set_password('admin123')
        admin.is_admin = True
        db.session.add(admin)
        db.session.commit()
        print("Default admin user created: username: admin / password: admin123")
    else:
        print("Admin user already exists.")


# ------------------------- Routes -------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Choose a different one.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please log in.', 'danger')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            age=int(request.form.get('age', 25) or 25),
            occupation=request.form.get('occupation', 'Student') or 'Student',
            monthly_income=float(request.form.get('monthly_income', 50000.0) or 50000.0),
            monthly_expenses=float(request.form.get('monthly_expenses', 35000.0) or 35000.0),
            savings=float(request.form.get('savings', 100000.0) or 100000.0),
            study_hours_per_week=float(request.form.get('study_hours_per_week', 20.0) or 20.0),
            fitness_hours_per_week=float(request.form.get('fitness_hours_per_week', 5.0) or 5.0),
            sleep_hours_per_day=float(request.form.get('sleep_hours_per_day', 7.0) or 7.0)
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        flash('Thank you! Your message has been sent successfully.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/privacy')
def privacy():
    priv_file = os.path.join(app.root_path, 'templates', 'privacy.html')
    if os.path.exists(priv_file):
        return render_template('privacy.html')
    return render_template_string("""
{% extends "base.html" %}
{% block content %}
<div class="container py-5">
    <div class="card p-4 rounded-4 shadow-sm">
        <h2 class="fw-bold mb-3"><i class="fas fa-shield-alt text-primary me-2"></i>Privacy Policy</h2>
        <p class="lead text-muted">Your data security and compliance privacy are fundamental priorities in our Intelligence Platform.</p>
        <p>All activity logs, simulation parameters, and digital twin models are kept strictly private and encrypted.</p>
    </div>
</div>
{% endblock %}
""")


@app.route('/terms')
def terms():
    terms_file = os.path.join(app.root_path, 'templates', 'terms.html')
    if os.path.exists(terms_file):
        return render_template('terms.html')
    return render_template_string("""
{% extends "base.html" %}
{% block content %}
<div class="container py-5">
    <div class="card p-4 rounded-4 shadow-sm">
        <h2 class="fw-bold mb-3"><i class="fas fa-file-contract text-primary me-2"></i>Terms of Service</h2>
        <p class="lead text-muted">Please review our visual risk intelligence and digital twin compliance usage guidelines.</p>
        <p>By using this platform, you agree to responsible modeling of financial, study, and wellness activities.</p>
    </div>
</div>
{% endblock %}
""")


# ------------------------- User Dashboard Route -------------------------
@app.route('/user/dashboard')
@login_required
def user_dashboard():
    # ===== Existing metrics =====
    metrics = calculate_financial_metrics(current_user)
    gpa_info = predict_gpa(current_user)
    total_fitness = sum(f.duration_min for f in current_user.fitness_logs)

    # ===== Chart Data Preparation =====
    monthly_income = defaultdict(float)
    monthly_expenses = defaultdict(float)
    for tx in current_user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_income[month_key] += tx.amount
        else:
            monthly_expenses[month_key] += tx.amount

    all_months = sorted(set(monthly_income.keys()) | set(monthly_expenses.keys()))
    months = all_months[-6:] if all_months else [datetime.now().strftime('%Y-%m')]
    income_data = [monthly_income.get(m, 0) for m in months]
    expense_data = [monthly_expenses.get(m, 0) for m in months]

    study_dates = defaultdict(float)
    for log in current_user.study_logs:
        study_dates[log.date.strftime('%Y-%m-%d')] += log.hours

    study_days = sorted(study_dates.keys())[-14:]
    study_hours = [study_dates.get(d, 0) for d in study_days]

    fitness_dates = defaultdict(float)
    for log in current_user.fitness_logs:
        fitness_dates[log.date.strftime('%Y-%m-%d')] += log.duration_min

    fitness_days = sorted(fitness_dates.keys())[-14:]
    fitness_minutes = [fitness_dates.get(d, 0) for d in fitness_days]

    goal_progress = []
    user_goals = getattr(current_user, 'goals', [])
    for goal in user_goals:
        if not getattr(goal, 'achieved', False) and getattr(goal, 'target_date', None):
            days_remaining = (goal.target_date - datetime.now().date()).days
            if days_remaining < 0:
                progress = 100
            else:
                progress = min(100, max(0, 100 - (days_remaining / 30 * 100)))
            goal_progress.append({'description': goal.description, 'progress': round(progress, 1)})
        elif getattr(goal, 'achieved', False):
            goal_progress.append({'description': goal.description, 'progress': 100})

    # ===== PREDICTIVE ANALYTICS =====
    monthly_net = defaultdict(float)
    for tx in current_user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_net[month_key] += tx.amount
        else:
            monthly_net[month_key] -= tx.amount

    sorted_months = sorted(monthly_net.keys())
    net_values = [monthly_net[m] for m in sorted_months[-6:]]
    if len(net_values) >= 3:
        forecast_net = generate_forecast(net_values, 6)
        savings_forecast = []
        cum = current_user.savings or 0.0
        for val in forecast_net:
            cum += val
            savings_forecast.append(round(cum, 2))
    else:
        current_savings = current_user.savings or 0.0
        savings_forecast = [round(current_savings + i * 5000, 2) for i in range(1, 7)]

    study_hours_by_month = defaultdict(float)
    for log in current_user.study_logs:
        month_key = log.date.strftime('%Y-%m')
        study_hours_by_month[month_key] += log.hours

    study_hours_list = [study_hours_by_month[m] for m in sorted(study_hours_by_month.keys())[-4:]]
    if len(study_hours_list) >= 3:
        forecast_hours = generate_forecast(study_hours_list, 3)
        gpa_forecast = []
        avg_prod = sum(l.productivity_score for l in current_user.study_logs) / len(current_user.study_logs) if current_user.study_logs else 70
        for hours in forecast_hours:
            gpa = 2.0 + (hours / 50) + (avg_prod / 200)
            gpa_forecast.append(round(min(4.0, max(1.0, gpa)), 2))
    else:
        gpa_forecast = [round(min(4.0, gpa_info['gpa'] + i * 0.05), 2) for i in range(1, 4)]

    fitness_by_day = defaultdict(float)
    for log in current_user.fitness_logs:
        fitness_by_day[log.date.strftime('%Y-%m-%d')] += log.duration_min

    fitness_days_list = list(fitness_by_day.values())[-14:]
    if len(fitness_days_list) >= 5:
        forecast_fitness = generate_forecast(fitness_days_list, 30)
        projected_fitness = round(sum(forecast_fitness), 1)
    else:
        projected_fitness = total_fitness * 2

    total_study_hours = sum(log.hours for log in current_user.study_logs) or 0

    study_streak = 0
    if current_user.study_logs:
        study_days_set = set(log.date for log in current_user.study_logs)
        today = datetime.now().date()
        for i in range(7):
            if (today - timedelta(days=i)) in study_days_set:
                study_streak += 1
            else:
                break

    fitness_streak = 0
    if current_user.fitness_logs:
        today = datetime.now().date()
        for i in range(7):
            day_total = sum(l.duration_min for l in current_user.fitness_logs if l.date == (today - timedelta(days=i)))
            if day_total >= 10:
                fitness_streak += 1
            else:
                break

    subject_totals = defaultdict(float)
    for log in current_user.study_logs:
        if log.subject:
            subject_totals[log.subject] += log.hours
    subjects = list(subject_totals.keys())
    subject_hours = list(subject_totals.values())

    activity_totals = defaultdict(float)
    for log in current_user.fitness_logs:
        activity_totals[log.activity] += log.duration_min
    activities = list(activity_totals.keys())
    activity_minutes = list(activity_totals.values())

    goals_count = len(user_goals)
    achieved_count = len([g for g in user_goals if getattr(g, 'achieved', False)])
    goal_ratio = (achieved_count / max(1, goals_count)) if goals_count > 0 else 0.5
    life_score = (
        (min(30.0, metrics['rate']) / 30.0) * 20.0 +
        (gpa_info['gpa'] / 4.0) * 30.0 +
        (min(total_fitness, 300) / 300.0) * 25.0 +
        goal_ratio * 25.0
    )
    life_score = max(0, min(100, round(life_score)))

    return render_template(
        'user/dashboard.html',
        user=current_user,
        metrics=metrics,
        gpa=gpa_info,
        total_fitness=total_fitness,
        months=months,
        income_data=income_data,
        expense_data=expense_data,
        study_days=study_days,
        study_hours=study_hours,
        fitness_days=fitness_days,
        fitness_minutes=fitness_minutes,
        goal_progress=goal_progress,
        savings_forecast=savings_forecast,
        gpa_forecast=gpa_forecast,
        projected_fitness=projected_fitness,
        total_study_hours=total_study_hours,
        study_streak=study_streak,
        fitness_streak=fitness_streak,
        subjects=subjects,
        subject_hours=subject_hours,
        activities=activities,
        activity_minutes=activity_minutes,
        life_score=life_score
    )


@app.route('/user/add_goal', methods=['POST'])
@login_required
def add_goal():
    description = request.form.get('description', '').strip()
    target_date_str = request.form.get('target_date', '')

    if not description:
        flash('Goal description is required.', 'danger')
        return redirect(url_for('user_dashboard'))

    target_date = None
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format.', 'danger')
            return redirect(url_for('user_dashboard'))

    goal = Goal(
        user_id=current_user.id,
        description=description,
        target_date=target_date,
        achieved=False
    )
    db.session.add(goal)
    db.session.commit()
    flash('Goal added successfully!', 'success')
    return redirect(url_for('user_dashboard'))


# ------------------------- Financial Management Routes -------------------------
@app.route('/user/financial', methods=['GET', 'POST'])
@app.route('/financial', methods=['GET', 'POST'])
@login_required
def user_financial():
    if request.method == 'POST':
        date_str = request.form.get('date')
        category = request.form.get('category', '').strip()
        amount_str = request.form.get('amount')
        trans_type = request.form.get('type')

        try:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            amount = float(amount_str)
        except (ValueError, TypeError):
            flash('Invalid date or amount specified.', 'danger')
            return redirect(url_for('user_financial'))

        tx = Transaction(user_id=current_user.id, date=date, category=category, amount=amount, type=trans_type)
        db.session.add(tx)
        db.session.commit()
        flash('Transaction recorded successfully!', 'success')
        return redirect(url_for('user_financial'))

    transactions = current_user.transactions
    expense_data = [t.amount for t in transactions if t.type == 'expense']
    forecast = generate_forecast(expense_data, 6) if expense_data else []
    metrics = calculate_financial_metrics(current_user)

    category_totals = {}
    for tx in transactions:
        if tx.type == 'expense':
            category_totals[tx.category] = category_totals.get(tx.category, 0) + tx.amount

    categories = list(category_totals.keys())
    category_amounts = list(category_totals.values())
    recent_transactions = transactions[-10:][::-1]

    return render_template(
        'user/financial.html',
        user=current_user,
        transactions=recent_transactions,
        forecast=forecast,
        metrics=metrics,
        categories=categories,
        category_amounts=category_amounts
    )


@app.route('/edit_transaction', methods=['POST'])
@login_required
def edit_transaction():
    tx_id = request.form.get('id')
    if not tx_id:
        flash('Transaction ID missing.', 'danger')
        return redirect(url_for('user_financial'))

    tx = Transaction.query.filter_by(id=tx_id, user_id=current_user.id).first()
    if not tx:
        flash('Transaction not found or you do not have permission.', 'danger')
        return redirect(url_for('user_financial'))

    try:
        tx.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        tx.category = request.form['category']
        tx.amount = float(request.form['amount'])
        tx.type = request.form['type']
        db.session.commit()
        flash('Transaction updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating transaction: {str(e)}', 'danger')

    return redirect(url_for('user_financial'))


@app.route('/delete_transaction/<int:id>', methods=['POST'])
@login_required
def delete_transaction(id):
    tx = Transaction.query.filter_by(id=id, user_id=current_user.id).first()
    if not tx:
        flash('Transaction not found or you do not have permission.', 'danger')
        return redirect(url_for('user_financial'))

    try:
        db.session.delete(tx)
        db.session.commit()
        flash('Transaction deleted.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting transaction: {str(e)}', 'danger')

    return redirect(url_for('user_financial'))


# ------------------------- Study Intelligence Routes -------------------------
@app.route('/user/study', methods=['GET', 'POST'])
@app.route('/study', methods=['GET', 'POST'])
@login_required
def user_study():
    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        hours = float(request.form['hours'])
        subject = request.form.get('subject', '').strip()
        productivity = int(request.form['productivity'])

        log = StudyLog(
            user_id=current_user.id,
            date=date,
            hours=hours,
            subject=subject,
            productivity_score=productivity
        )
        db.session.add(log)
        db.session.commit()
        flash('Study session logged successfully!', 'success')
        return redirect(url_for('user_study'))

    logs = current_user.study_logs
    gpa_info = predict_gpa(current_user)

    study_by_day = defaultdict(float)
    for log in logs:
        study_by_day[log.date.strftime('%Y-%m-%d')] += log.hours

    study_dates = sorted(study_by_day.keys())[-14:]
    study_hours = [study_by_day.get(d, 0) for d in study_dates]

    subject_totals = defaultdict(float)
    for log in logs:
        if log.subject:
            subject_totals[log.subject] += log.hours

    subjects = list(subject_totals.keys())
    subject_hours = list(subject_totals.values())

    prod_by_day = defaultdict(list)
    for log in logs:
        prod_by_day[log.date.strftime('%Y-%m-%d')].append(log.productivity_score)

    prod_dates = sorted(prod_by_day.keys())[-14:]
    prod_avg = [sum(prod_by_day[d]) / len(prod_by_day[d]) if d in prod_by_day else 0 for d in prod_dates]

    total_hours = sum(log.hours for log in logs)
    avg_productivity = sum(log.productivity_score for log in logs) / len(logs) if logs else 0

    study_streak = 0
    if logs:
        study_days_set = sorted(set(log.date for log in logs), reverse=True)
        today = datetime.now().date()
        for i in range(7):
            check_date = today - timedelta(days=i)
            if check_date in study_days_set:
                study_streak += 1
            else:
                break

    recent_logs = logs[-10:][::-1]

    return render_template(
        'user/study.html',
        user=current_user,
        logs=recent_logs,
        gpa=gpa_info,
        study_dates=study_dates,
        study_hours=study_hours,
        subjects=subjects,
        subject_hours=subject_hours,
        prod_dates=prod_dates,
        prod_avg=prod_avg,
        total_hours=total_hours,
        avg_productivity=avg_productivity,
        study_streak=study_streak
    )


@app.route('/edit_study_log', methods=['POST'])
@login_required
def edit_study_log():
    log_id = request.form.get('id')
    if not log_id:
        flash('Log ID missing.', 'danger')
        return redirect(url_for('user_study'))

    log = StudyLog.query.filter_by(id=log_id, user_id=current_user.id).first()
    if not log:
        flash('Log not found or you do not have permission.', 'danger')
        return redirect(url_for('user_study'))

    try:
        log.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        log.hours = float(request.form['hours'])
        log.subject = request.form.get('subject')
        log.productivity_score = int(request.form['productivity'])
        db.session.commit()
        flash('Study log updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating log: {str(e)}', 'danger')

    return redirect(url_for('user_study'))


@app.route('/delete_study_log/<int:id>', methods=['POST'])
@login_required
def delete_study_log(id):
    log = StudyLog.query.filter_by(id=id, user_id=current_user.id).first()
    if not log:
        flash('Log not found or you do not have permission.', 'danger')
        return redirect(url_for('user_study'))

    try:
        db.session.delete(log)
        db.session.commit()
        flash('Study log deleted successfully!', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting log: {str(e)}', 'danger')

    return redirect(url_for('user_study'))


# ------------------------- Fitness Tracking Routes -------------------------
@app.route('/user/fitness', methods=['GET', 'POST'])
@app.route('/fitness', methods=['GET', 'POST'])
@login_required
def user_fitness():
    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        activity = request.form['activity']
        duration = int(request.form['duration'])
        calories = int(request.form['calories'])

        log = FitnessLog(
            user_id=current_user.id,
            date=date,
            activity=activity,
            duration_min=duration,
            calories_burned=calories
        )
        db.session.add(log)
        db.session.commit()
        flash('Fitness activity logged successfully!', 'success')
        return redirect(url_for('user_fitness'))

    logs = current_user.fitness_logs
    total_min = sum(l.duration_min for l in logs)
    total_cal = sum(l.calories_burned or 0 for l in logs)

    days_since_first = (datetime.now().date() - (logs[0].date if logs else datetime.now().date())).days
    avg_min_per_day = total_min / max(1, days_since_first if days_since_first > 0 else 1)

    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())
    weekly_logs = [l for l in logs if l.date >= start_of_week]
    weekly_min = sum(l.duration_min for l in weekly_logs)
    week_goal = 150
    week_progress = min(100, (weekly_min / week_goal) * 100)

    daily_totals = defaultdict(int)
    for log in logs:
        daily_totals[log.date.strftime('%Y-%m-%d')] += log.duration_min

    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(13, -1, -1)]
    daily_data = [daily_totals.get(d, 0) for d in dates]

    activity_totals = defaultdict(int)
    for log in logs:
        activity_totals[log.activity] += log.duration_min
    activities = list(activity_totals.keys())
    activity_minutes = list(activity_totals.values())

    streak = 0
    if logs:
        for i in range(30):
            check_date = today - timedelta(days=i)
            day_total = sum(l.duration_min for l in logs if l.date == check_date)
            if day_total >= 10:
                streak += 1
            else:
                break

    recent_logs = logs[-10:][::-1]

    return render_template(
        'user/fitness.html',
        user=current_user,
        logs=recent_logs,
        total_min=total_min,
        total_cal=total_cal,
        avg_min_per_day=round(avg_min_per_day, 1),
        weekly_min=weekly_min,
        week_progress=round(week_progress, 1),
        dates=dates,
        daily_data=daily_data,
        activities=activities,
        activity_minutes=activity_minutes,
        streak=streak
    )


@app.route('/fitness/edit/<int:log_id>', methods=['POST'])
@login_required
def edit_fitness(log_id):
    log = FitnessLog.query.get_or_404(log_id)
    if log.user_id != current_user.id:
        flash('You are not authorised to edit this activity.', 'danger')
        return redirect(url_for('user_fitness'))

    date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
    activity = request.form['activity']
    duration = int(request.form['duration'])
    calories = int(request.form['calories'])

    log.date = date
    log.activity = activity
    log.duration_min = duration
    log.calories_burned = calories
    db.session.commit()
    flash('Activity updated successfully!', 'success')
    return redirect(url_for('user_fitness'))


@app.route('/fitness/delete/<int:log_id>', methods=['POST'])
@login_required
def delete_fitness(log_id):
    log = FitnessLog.query.get_or_404(log_id)
    if log.user_id != current_user.id:
        flash('You are not authorised to delete this activity.', 'danger')
        return redirect(url_for('user_fitness'))

    db.session.delete(log)
    db.session.commit()
    flash('Activity deleted successfully!', 'info')
    return redirect(url_for('user_fitness'))


# Milestone Placeholders for Navigation Consistency
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    flash('Admin Analytics Dashboard is scheduled for future milestones.', 'info')
    return redirect(url_for('user_dashboard'))


@app.route('/admin/users')
@login_required
def admin_users():
    flash('Admin User Management is scheduled for future milestones.', 'info')
    return redirect(url_for('user_dashboard'))


@app.route('/user/simulation')
@login_required
def user_simulation():
    flash('Future Simulation Engine is scheduled for Milestone 5.', 'info')
    return redirect(url_for('user_dashboard'))


@app.route('/user/assistant')
@login_required
def user_assistant():
    flash('AI Assistant is scheduled for Milestone 6.', 'info')
    return redirect(url_for('user_dashboard'))


@app.route('/settings')
@login_required
def settings():
    flash('Settings & Theme configuration.', 'info')
    return redirect(url_for('user_dashboard'))


# ------------------------- Run the App -------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_default_admin()
    app.run(debug=True)

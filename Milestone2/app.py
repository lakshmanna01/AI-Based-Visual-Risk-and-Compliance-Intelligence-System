import os
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from sklearn.linear_model import LinearRegression

load_dotenv()

app = Flask(__name__, template_folder='Templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///digital_twin.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
        print("Default admin user created:")
        print("Username: admin")
        print("Password: admin123")
        print("Please change your password after first login.")
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
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('All fields are required.')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters.')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username and password.')
            return render_template('login.html')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.username}! (Milestone 2 Active: Financial Management)')
            return redirect(url_for('user_financial'))

        flash('Invalid username or password.')
        return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('index'))


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()

        if not all([name, email, subject, message]):
            flash('All fields are required.')
            return redirect(url_for('contact'))

        flash("Your message has been sent successfully! We'll get back to you soon.")
        return redirect(url_for('contact'))

    return render_template('contact.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


# ------------------------- Milestone 2 Financial Routes -------------------------
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
        'financial.html',
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


# Milestone 3 & Beyond Placeholders
@app.route('/user/dashboard')
@login_required
def user_dashboard():
    return redirect(url_for('user_financial'))


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    flash('Admin Analytics Dashboard is scheduled for future milestones.')
    return redirect(url_for('user_financial'))


@app.route('/admin/users')
@login_required
def admin_users():
    flash('Admin User Management is scheduled for future milestones.')
    return redirect(url_for('user_financial'))


@app.route('/user/study')
@login_required
def user_study():
    flash('Study Intelligence is scheduled for Milestone 3.')
    return redirect(url_for('user_financial'))


@app.route('/user/fitness')
@login_required
def user_fitness():
    flash('Fitness Tracking is scheduled for Milestone 4.')
    return redirect(url_for('user_financial'))


@app.route('/user/simulation')
@login_required
def user_simulation():
    flash('Future Simulation Engine is scheduled for Milestone 5.')
    return redirect(url_for('user_financial'))


@app.route('/user/assistant')
@login_required
def user_assistant():
    flash('AI Assistant is scheduled for Milestone 6.')
    return redirect(url_for('user_financial'))


@app.route('/settings')
@login_required
def settings():
    flash('Settings & Themes.')
    return redirect(url_for('user_financial'))


@app.route('/about')
def about():
    flash('Visual Risk & Compliance Intelligence System — Milestone 2 active.')
    return redirect(url_for('index'))


# ------------------------- Run the App -------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_default_admin()
    app.run(debug=True)

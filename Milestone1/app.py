import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

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

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


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
            flash(f'Welcome back, {user.username}! (Milestone 1 Active: Core Auth & Landing Pages)')
            return redirect(url_for('index'))

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


# Milestone 2 Feature Placeholders (keeps all navbar links functional in Milestone 1)
@app.route('/user/dashboard')
@login_required
def user_dashboard():
    flash('User Dashboard analytics will be introduced in Milestone 2.')
    return redirect(url_for('index'))


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    flash('Admin Analytics Dashboard is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/admin/users')
@login_required
def admin_users():
    flash('Admin User Management is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/user/financial')
@login_required
def user_financial():
    flash('Financial Management is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/user/study')
@login_required
def user_study():
    flash('Study Intelligence is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/user/fitness')
@login_required
def user_fitness():
    flash('Fitness Tracking is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/user/simulation')
@login_required
def user_simulation():
    flash('Simulation Engine is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/user/assistant')
@login_required
def user_assistant():
    flash('AI Assistant is scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/settings')
@login_required
def settings():
    flash('Theme and Profile settings are scheduled for Milestone 2.')
    return redirect(url_for('index'))


@app.route('/about')
def about():
    flash('Visual Risk & Compliance Intelligence System — Milestone 1 active.')
    return redirect(url_for('index'))


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')



# ------------------------- Run the App -------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_default_admin()
    app.run(debug=True)

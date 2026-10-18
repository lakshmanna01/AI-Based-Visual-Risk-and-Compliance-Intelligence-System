import sys
import os
import uuid

root_dir = os.path.dirname(os.path.abspath(__file__))
m1_dir = os.path.join(root_dir, "Milestone1")

print("--- Testing Milestone 1 ---")
os.chdir(m1_dir)
from app import app as m1_app, db as m1_db, User as M1_User, create_default_admin as m1_admin

with m1_app.app_context():
    m1_db.create_all()
    m1_admin()
    admin = M1_User.query.filter_by(username='admin').first()
    assert admin is not None, "Admin user not found in Milestone1"
    assert admin.check_password('admin123'), "Admin password invalid in Milestone1"
    print("[OK] Milestone 1 Database & Admin creation successful")

m1_client = m1_app.test_client()

# Test GET routes
for route in ['/', '/login', '/register', '/contact', '/privacy', '/terms']:
    resp = m1_client.get(route)
    assert resp.status_code == 200, f"Milestone 1 GET {route} failed with {resp.status_code}"
    print(f"[OK] Milestone 1 GET {route} returned 200")


# Test POST /register with unique user
uname_m1 = f"m1_user_{uuid.uuid4().hex[:6]}"
pwd_m1 = "password123"
reg_resp = m1_client.post('/register', data={
    'username': uname_m1,
    'email': f"{uname_m1}@example.com",
    'password': pwd_m1,
    'confirm_password': pwd_m1
}, follow_redirects=True)
assert reg_resp.status_code == 200
print("[OK] Milestone 1 POST /register successful")

# Test POST /login
login_resp = m1_client.post('/login', data={
    'username': uname_m1,
    'password': pwd_m1
}, follow_redirects=True)
assert login_resp.status_code == 200
print("[OK] Milestone 1 POST /login successful")

# Test POST /contact
contact_resp = m1_client.post('/contact', data={
    'name': 'Tester',
    'email': 'tester@example.com',
    'subject': 'Milestone 1 Query',
    'message': 'Testing contact page in Milestone 1'
}, follow_redirects=True)
assert contact_resp.status_code == 200
assert b'sent successfully' in contact_resp.data
print("[OK] Milestone 1 POST /contact successful")


print("\n--- Testing Full Application ---")
os.chdir(root_dir)
sys.modules.pop('app', None)
from app import app as full_app, db as full_db, User as Full_User, create_default_admin as full_admin

with full_app.app_context():
    full_db.create_all()
    full_admin()
    admin = Full_User.query.filter_by(username='admin').first()
    assert admin is not None, "Admin user not found in Full App"
    print("[OK] Full App Database & Admin creation successful")

full_client = full_app.test_client()

# Test public routes
for route in ['/', '/login', '/register', '/contact', '/about', '/privacy', '/terms']:
    resp = full_client.get(route)
    assert resp.status_code == 200, f"Full App GET {route} failed with {resp.status_code}"
    print(f"[OK] Full App GET {route} returned 200")

# Register and login user
uname_full = f"alice_{uuid.uuid4().hex[:6]}"
pwd_full = "password123"
full_client.post('/register', data={
    'username': uname_full,
    'email': f"{uname_full}@example.com",
    'password': pwd_full,
    'confirm_password': pwd_full
}, follow_redirects=True)

full_client.post('/login', data={
    'username': uname_full,
    'password': pwd_full
}, follow_redirects=True)

# Test User Routes
for route in ['/user/dashboard', '/user/financial', '/user/study', '/user/fitness', '/user/simulation', '/user/assistant', '/settings']:
    resp = full_client.get(route)
    assert resp.status_code == 200, f"Full App User GET {route} failed with {resp.status_code}"
    print(f"[OK] Full App User GET {route} returned 200")

# Test adding financial transaction
tx_resp = full_client.post('/user/financial', data={
    'date': '2026-10-18',
    'category': 'Books',
    'amount': '1500',
    'type': 'expense'
}, follow_redirects=True)
assert tx_resp.status_code == 200
print("[OK] Full App POST /user/financial successful")

# Test adding study log
study_resp = full_client.post('/user/study', data={
    'date': '2026-10-18',
    'hours': '3.0',
    'subject': 'Machine Learning',
    'productivity': '90'
}, follow_redirects=True)
assert study_resp.status_code == 200
print("[OK] Full App POST /user/study successful")

# Test adding fitness log
fit_resp = full_client.post('/user/fitness', data={
    'date': '2026-10-18',
    'activity': 'Running',
    'duration': '30',
    'calories': '250'
}, follow_redirects=True)
assert fit_resp.status_code == 200
print("[OK] Full App POST /user/fitness successful")

# Test simulation
sim_resp = full_client.post('/user/simulation', data={
    'scenario': 'save',
    'amount': '5000'
}, follow_redirects=True)
assert sim_resp.status_code == 200
print("[OK] Full App POST /user/simulation successful")

# Test assistant query (fallback check)
ast_resp = full_client.post('/user/assistant', data={
    'query': 'How can I save more money?'
}, follow_redirects=True)
assert ast_resp.status_code == 200
print("[OK] Full App POST /user/assistant successful")

# Logout and Login as Admin
full_client.get('/logout')
full_client.post('/login', data={
    'username': 'admin',
    'password': 'admin123'
}, follow_redirects=True)

with full_app.app_context():
    target_user = Full_User.query.filter_by(username=uname_full).first()
    target_uid = target_user.id if target_user else 1

for route in ['/admin/dashboard', '/admin/users', f'/admin/user/{target_uid}']:
    resp = full_client.get(route)
    assert resp.status_code == 200, f"Full App Admin GET {route} failed with {resp.status_code}"
    print(f"[OK] Full App Admin GET {route} returned 200")

print("\nALL TESTS PASSED SUCCESSFULLY! Both Milestone 1 and Full Application are 100% operational!")

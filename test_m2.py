import os
import sys
import uuid

root = os.path.dirname(os.path.abspath(__file__))
m2_dir = os.path.join(root, "Milestone 2", "Ai-based-visual-risk")
if not os.path.exists(m2_dir):
    m2_dir = os.path.join(root, "Milestone2")
os.chdir(m2_dir)
sys.path.insert(0, m2_dir)

from app import app, db, User, Transaction, create_default_admin

with app.app_context():
    db.create_all()
    create_default_admin()
    admin = User.query.filter_by(username='admin').first()
    assert admin is not None
    print("[OK] Milestone 2 Admin initialized")

client = app.test_client()

# Test public routes
for route in ['/', '/login', '/register', '/contact', '/privacy', '/terms']:
    resp = client.get(route)
    assert resp.status_code == 200, f"Failed GET {route}"
    print(f"[OK] Milestone 2 GET {route} -> 200")

# Register and login user
uname = f"m2_user_{uuid.uuid4().hex[:6]}"
pwd = "password123"

client.post('/register', data={
    'username': uname,
    'email': f"{uname}@example.com",
    'password': pwd,
    'confirm_password': pwd
}, follow_redirects=True)

login_resp = client.post('/login', data={
    'username': uname,
    'password': pwd
}, follow_redirects=True)
assert login_resp.status_code == 200
print("[OK] Milestone 2 Register & Login successful")

# Test GET /user/financial
fin_resp = client.get('/user/financial')
assert fin_resp.status_code == 200
assert b'Milestone 2: Financial Management' in fin_resp.data
print("[OK] Milestone 2 GET /user/financial -> 200 (renders financial.html)")

# Test POST /user/financial (add income)
inc_resp = client.post('/user/financial', data={
    'date': '2026-10-18',
    'category': 'Salary',
    'amount': '75000',
    'type': 'income'
}, follow_redirects=True)
assert inc_resp.status_code == 200
assert b'Transaction recorded successfully' in inc_resp.data
print("[OK] Milestone 2 POST /user/financial (income) successful")

# Test POST /user/financial (add expense)
exp_resp = client.post('/user/financial', data={
    'date': '2026-10-18',
    'category': 'Groceries',
    'amount': '12500',
    'type': 'expense'
}, follow_redirects=True)
assert exp_resp.status_code == 200
print("[OK] Milestone 2 POST /user/financial (expense) successful")

with app.app_context():
    user = User.query.filter_by(username=uname).first()
    assert len(user.transactions) == 2
    tx_id = user.transactions[-1].id

# Test edit transaction
edit_resp = client.post('/edit_transaction', data={
    'id': tx_id,
    'date': '2026-10-18',
    'category': 'Organic Groceries',
    'amount': '15000',
    'type': 'expense'
}, follow_redirects=True)
assert edit_resp.status_code == 200
assert b'Transaction updated successfully' in edit_resp.data
print("[OK] Milestone 2 POST /edit_transaction successful")

# Test delete transaction
del_resp = client.post(f'/delete_transaction/{tx_id}', follow_redirects=True)
assert del_resp.status_code == 200
assert b'Transaction deleted' in del_resp.data
print("[OK] Milestone 2 POST /delete_transaction successful")

print("\nALL MILESTONE 2 TESTS PASSED PERFECTLY!")

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple user class
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username
        self.user_folder = os.path.join('users', username)

# Dictionary to store users (replace with database in production)
users = {}

@login_manager.user_loader
def load_user(username):
    return users.get(username)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully.')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))
        
        # Create user folder
        user_folder = os.path.join('users', username)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        # Create new user
        user = User(username)
        user.password_hash = generate_password_hash(password, method='pbkdf2:sha256')  # Using pbkdf2 with sha256
        users[username] = user
        
        # Store registration information in a file
        info_file = os.path.join(user_folder, 'user_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Username: {username}\n")
            f.write(f"Password Hash: {user.password_hash}\n")
            f.write(f"Registration Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create users directory if it doesn't exist
    if not os.path.exists('users'):
        os.makedirs('users')
    
    app.run(debug=True, port=5001)
# app.py
# Final version with the database initialization fix for production servers.

import joblib
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, make_response
import os
import sqlite3
from datetime import datetime
import math
import io
import csv

# --- Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
ADMIN_PASSWORD = os.environ.get('SPAM_ADMIN_PASSWORD', 'admin')
DATABASE = 'predictions.db'

# --- Database & Model Loading ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # This function now uses app.app_context() to work correctly on startup.
    with app.app_context():
        conn = get_db_connection()
        cursor = conn.cursor()
        if not cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone():
            print("--- Creating 'predictions' table ---")
            conn.execute('''
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    spam_probability REAL NOT NULL
                );
            ''')
            conn.commit()
        else:
            print("--- 'predictions' table already exists ---")
        conn.close()

try:
    model = joblib.load('spam_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
except FileNotFoundError:
    model, vectorizer = None, None

# ----- THE FIX IS HERE -----
# We call the database initialization function right after the app is created.
# This ensures it runs every time, even when started by Gunicorn on Render.
init_db()
# ---------------------------

# --- Public & Admin Routes (No changes below this line) ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model is not loaded.'}), 500
    data = request.get_json()
    if not data or 'email_text' not in data:
        return jsonify({'error': 'Invalid input.'}), 400
    email_text = data['email_text']
    email_tfidf = vectorizer.transform([email_text])
    prediction_result = model.predict(email_tfidf)[0]
    spam_prob_float = model.predict_proba(email_tfidf)[0][1]
    try:
        conn = get_db_connection()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            'INSERT INTO predictions (timestamp, input_text, prediction, spam_probability) VALUES (?, ?, ?, ?)',
            (timestamp, email_text, prediction_result.upper(), spam_prob_float)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
    return jsonify({
        'prediction': prediction_result.upper(),
        'spam_probability': f"{spam_prob_float:.2%}"
    })

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Incorrect password.', 'error')
    return render_template('admin_login.html')

@app.route('/admin')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    records = conn.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    total_predictions = len(records)
    spam_count = conn.execute("SELECT COUNT(id) FROM predictions WHERE prediction = 'SPAM'").fetchone()[0]
    conn.close()
    spam_rate = (spam_count / total_predictions * 100) if total_predictions > 0 else 0
    stats = { 'total_predictions': total_predictions, 'spam_rate': math.floor(spam_rate) }
    return render_template('admin_history.html', records=records, stats=stats)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin/download/<format>')
def download_report(format):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    conn = get_db_connection()
    records = conn.execute('SELECT timestamp, input_text, prediction, spam_probability FROM predictions ORDER BY id DESC').fetchall()
    conn.close()
    if format == 'csv':
        string_io = io.StringIO()
        csv_writer = csv.writer(string_io)
        csv_writer.writerow(['Timestamp', 'Input Text', 'Prediction', 'Spam Probability'])
        for record in records:
            csv_writer.writerow([
                record['timestamp'], record['input_text'], record['prediction'],
                f"{record['spam_probability']:.2%}"
            ])
        output = make_response(string_io.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    return "Invalid format", 404

# --- Main Execution (Now only used for local development) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


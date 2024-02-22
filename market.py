import subprocess
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/run_python_script', methods=['POST'])
def run_python_script():
        result = subprocess.run(['python', 'C:/Users/hp/PycharmProjects/handtracking/VolumeHandControlAdvanced.py'], capture_output=True, text=True)
        
if __name__ == '__main__':
    app.run(debug=True)
from flask import Blueprint, render_template, request
from .model import predict

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/detect-fraud', methods=['POST'])
def detect_fraud():
    data = request.json
    return predict(data)

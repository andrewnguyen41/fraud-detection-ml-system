from flask import Flask
from .model import ModelCache
from flask_apscheduler import APScheduler
from apscheduler.triggers.interval import IntervalTrigger
from .model import check_for_new_data
from dotenv import load_dotenv

def create_app():
    app = Flask(__name__)
    load_dotenv()

    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()

    # Schedule the check_for_new_file function to run every hour
    scheduler.add_job(
        func=check_for_new_data,
        trigger=IntervalTrigger(hours=1),
        id='check_new_data_files',
        name='Check for new data files in S3 every hour',
        replace_existing=True,
    )

    from .route import bp as main_bp
    app.register_blueprint(main_bp)

    return app

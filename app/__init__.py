from flask import Flask
# from app.models import db
from config import BaseConfig
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object(BaseConfig)
db = SQLAlchemy(app)

from app.models import *

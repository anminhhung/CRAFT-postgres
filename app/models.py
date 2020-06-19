import datetime
from flask_sqlalchemy import SQLAlchemy
from app import db

# db = SQLAlchemy()

class BaseModel(db.Model):
    """Base data model for all objects"""
    __abstract__ = True

    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        """Define a base way to print models"""
        return '%s(%s)' % (self.__class__.__name__, {
            column: value
            for column, value in self._to_dict().items()
        })

    def json(self):
        """
                Define a base way to jsonify models, dealing with datetime objects
        """
        return {
            column: value if not isinstance(value, datetime.date) else value.strftime('%Y-%m-%d')
            for column, value in self._to_dict().items()
        }

class Photo(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    link = db.Column(db.String(200), index=True, unique=True) 
    resultPhotos = db.relationship('CraftPhoto', backref='root', lazy='dynamic')

    def __repr__(self):
        return '<Link of Photo: {}>'.format(self.link)

class CraftPhoto(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    link = db.Column(db.String(200), index=True, unique=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.datetime.utcnow)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'))

    def __repr__(self):
        return '<Link of craftPhoto: {}>'.format(self.link)

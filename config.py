import os

class BaseConfig(object):
    SECRET_KEY = 'my-secret-key'
    DEBUG = False
    DB_NAME = 'craft'
    DB_USER = 'postgres'
    DB_PASS = 'postgres'
    DB_SERVICE = 'db'
    DB_PORT = '5432'
    SQLALCHEMY_DATABASE_URI = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(
        DB_USER, DB_PASS, DB_SERVICE, DB_PORT, DB_NAME
    )
import configparser
import pymysql

config = configparser.ConfigParser()
config.read("db.conf",encoding='utf-8')
db = config['database']
dbhost = db['dbhost']
dbport = db['dbport']
dbuser = db['user']
dbpw = db['password']
dbname = db['dbname']

def get_connection(database=dbname):
    conn = pymysql.connect(
        host=dbhost,
        user=dbuser, password=dbpw,
        database=database, charset="utf8")
    return conn

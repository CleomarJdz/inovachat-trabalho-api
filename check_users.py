from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:120206@localhost:5432/inovachat"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    senha = db.Column(db.String(255))

try:
    with app.app_context():
        usuarios = Usuario.query.all()
        print(f"Total de usuários no banco: {len(usuarios)}")
        if usuarios:
            print("\nUsuários cadastrados:")
            for u in usuarios:
                print(f"  ID: {u.id}, Nome: {u.nome}, Email: {u.email}")
        else:
            print("Nenhum usuário encontrado no banco de dados.")
except Exception as e:
    print(f"Erro ao consultar banco: {e}")

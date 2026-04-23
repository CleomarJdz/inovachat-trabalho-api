from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import google.generativeai as genai
import markdown
import os


app = Flask(__name__)
app.config["SECRET_KEY"] = "123456"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///inovachat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCusNeFs26xllqT7F-9lfsUBarfd4l4RBg')
genai.configure(api_key=GEMINI_API_KEY)

@app.template_filter('markdown')
def markdown_filter(text):
    return markdown.markdown(
        text, 
        extensions=['nl2br', 'fenced_code', 'tables', 'codehilite']
    ) if text else ''

class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    senha = db.Column(db.String(255))


class Conversa(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer)
    pergunta = db.Column(db.Text)
    resposta = db.Column(db.Text)
    data = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

def responder_ia(pergunta):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        resposta = model.generate_content(pergunta)
        texto = getattr(resposta, 'text', None)
        if texto:
            return texto
        return 'Sem resposta de IA'

    except Exception as erro:
        return f'Erro IA: {erro}'

@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/cadastro", methods=["GET", "POST"])
def cadastro():

    if request.method == "POST":

        nome = request.form["nome"]
        email = request.form["email"]
        senha = request.form["senha"]

        usuario_existente = Usuario.query.filter_by(email=email).first()

        if usuario_existente:
            flash("Email já cadastrado!")
            return redirect(url_for("cadastro"))

        senha_hash = generate_password_hash(senha)

        novo = Usuario(
            nome=nome,
            email=email,
            senha=senha_hash
        )

        db.session.add(novo)
        db.session.commit()

        flash("Cadastro realizado com sucesso!")
        return redirect(url_for("login"))

    return render_template("cadastro.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        senha = request.form["senha"]

        usuario = Usuario.query.filter_by(email=email).first()

        if usuario and check_password_hash(usuario.senha, senha):

            session["user_id"] = usuario.id
            session["nome"] = usuario.nome

            return redirect(url_for("chat"))

        flash("Login inválido!")

    return render_template("login.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        pergunta = request.form["pergunta"]
        resposta = responder_ia(pergunta)

        nova = Conversa(
            usuario_id=session["user_id"],
            pergunta=pergunta,
            resposta=resposta
        )

        db.session.add(nova)
        db.session.commit()

    historico = Conversa.query.filter_by(
        usuario_id=session["user_id"]
    ).order_by(Conversa.id.desc()).all()

    return render_template("chat.html", historico=historico)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
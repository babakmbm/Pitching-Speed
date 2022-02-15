import os
from flask import Flask, render_template, url_for, redirect, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug.utils import redirect
from wtforms import StringField, PasswordField, BooleanField, SelectField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from ObjectTracking import ObjectTracking
from DataPreperation_yolo import DataPrepration

app = Flask(__name__)
app.config['SECRET_KEY'] = 'DASSECRETKEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///D:\Students\Das\Baseball Analysis\Pitching Speed\ball.db'
Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
data_prep = DataPrepration()
UPLOAD_PATH = 'D:\Students\Das\Baseball Analysis\Pitching Speed\static\\upload\\'
print(UPLOAD_PATH)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(15))
    lastname = db.Column(db.String(15))
    username = db.Column(db.String(15), unique=True)
    password = db.Column(db.String(100))
    repeat_password = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(15))
    email_address = db.Column(db.String(150))
    phone_number = db.Column(db.String(20), unique=True)
    role = db.Column(db.String(15))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=5, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember_me')


class SignupForm(FlaskForm):
    firstname = StringField('firstname', validators=[InputRequired(), Length(min=3, max=15)])
    lastname = StringField('lastname', validators=[InputRequired(), Length(min=3, max=15)])
    username = StringField('username', validators=[InputRequired(), Length(min=5, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=100)])
    repeat_password = PasswordField('repeat_password', validators=[InputRequired(), Length(min=8, max=80)])
    age = StringField('age', validators=[InputRequired(), Length(min=1, max=3)])
    gender = StringField('gender', validators=[InputRequired(), Length(min=4, max=15)])
    email_address = StringField('email_address', validators=[InputRequired(), Email(message='Invalid Email')])
    phone_number = StringField('phone_number', validators=[InputRequired(), Length(min=5, max=20)])
    role = StringField('role', validators=[InputRequired(), Length(min=5, max=10)])


class UploadForm(FlaskForm):
    file_upload = FileField('file_upload')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('index'))

        return '<h1>Invalid Username or Password!</h1>'

    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        hashed_re_password = generate_password_hash(form.repeat_password.data, method='sha256')
        new_user = User(firstname=form.firstname.data, lastname=form.lastname.data, username=form.username.data,
                        password=hashed_password, repeat_password=hashed_re_password, age=form.age.data,
                        gender=form.gender.data, email_address=form.email_address.data,
                        phone_number=form.phone_number.data,
                        role=form.role.data)
        db.session.add(new_user)
        db.session.commit()
        return '<h1> A new user has been added to the database </h1>'

    return render_template('signup.html', form=form)


@app.route('/pitch', methods=['GET', 'POST'])
def pitch():
    form = UploadForm()
    if request.method == 'POST':
        upload_file = request.files['file_upload']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        # print(path_save)
        upload_file.save(path_save)
        # tracking = ObjectTracking("Boosting", path_save)
        # tracking.track_video_single()
        return render_template('pitch.html', form=form, filename=filename, upload=True)

    return render_template('pitch.html', form=form, upload=False)

@app.route('/labeling', methods=['GET', 'POST'])
def labeling():
    if request.method == 'POST':
        data_prep.label_images()
    return render_template('image_labeling.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)

# database.py — SQLAlchemy models for FinanceAI
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(120), nullable=False)
    email      = db.Column(db.String(200), unique=True, nullable=False)
    password   = db.Column(db.String(255), nullable=False)
    is_admin   = db.Column(db.Boolean, default=False)
    city       = db.Column(db.String(100), default='Other')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    analyses = db.relationship('FinanceAnalysis', backref='user', lazy=True,
                               cascade='all, delete-orphan')
    goals    = db.relationship('FinancialGoal',   backref='user', lazy=True,
                               cascade='all, delete-orphan')

    def set_password(self, raw):
        self.password = generate_password_hash(raw)

    def check_password(self, raw):
        return check_password_hash(self.password, raw)

    def __repr__(self):
        return f'<User {self.email}>'


class FinancialGoal(db.Model):
    __tablename__ = 'financial_goals'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    goal_type       = db.Column(db.String(100), nullable=False)
    goal_label      = db.Column(db.String(200), nullable=False)
    target_amount   = db.Column(db.Float, nullable=False)
    duration_months = db.Column(db.Integer, nullable=False)
    monthly_target  = db.Column(db.Float, nullable=False)
    is_active       = db.Column(db.Boolean, default=True)
    skipped         = db.Column(db.Boolean, default=False)

    # NEW: status field — 'active' | 'completed' | 'cancelled'
    status          = db.Column(db.String(20), default='active', nullable=False)
    # NEW: soft-delete / audit timestamps
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at    = db.Column(db.DateTime, nullable=True)

    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    analyses = db.relationship('FinanceAnalysis', backref='goal', lazy=True)

    @property
    def is_cancelled(self):
        return self.status == 'cancelled'

    @property
    def is_completed(self):
        return self.status == 'completed'

    def soft_cancel(self):
        """Soft-delete: mark cancelled but keep the DB row for history."""
        self.status       = 'cancelled'
        self.is_active    = False
        self.cancelled_at = datetime.utcnow()

    def __repr__(self):
        return f'<Goal {self.goal_label} status={self.status} user={self.user_id}>'


class FinanceAnalysis(db.Model):
    __tablename__ = 'finance_analyses'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    goal_id         = db.Column(db.Integer, db.ForeignKey('financial_goals.id'), nullable=True)
    income          = db.Column(db.Float, nullable=False)
    expenses        = db.Column(db.JSON, nullable=False)
    health_score    = db.Column(db.Float, nullable=True)
    cluster_label   = db.Column(db.String(100), nullable=True)
    recommendations = db.Column(db.JSON, nullable=True)
    goal_progress   = db.Column(db.Float, nullable=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Analysis income={self.income} user={self.user_id}>'

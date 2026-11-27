"""SQLAlchemy models + session handling (SQLite by default, PostgreSQL via DATABASE_URL)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from . import config


def _now() -> datetime:
    return datetime.now(timezone.utc)


_is_sqlite = config.DATABASE_URL.startswith("sqlite")
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=not _is_sqlite,
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):  # enforce FKs / cascades in SQLite
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    genre: Mapped[str] = mapped_column(String(40), default="general")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    versions: Mapped[list["Version"]] = relationship(back_populates="episode", cascade="all, delete-orphan",
                                                     order_by="Version.version_number")


class Version(Base):
    """One analyzed draft (script and/or video) of an episode."""
    __tablename__ = "versions"
    __table_args__ = (UniqueConstraint("episode_id", "version_number"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    genre: Mapped[str] = mapped_column(String(40), default="general")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    script_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    engine: Mapped[str] = mapped_column(String(40))                 # heuristic | llm | hybrid
    hook_score: Mapped[int] = mapped_column(Integer)
    ttfc_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    pacing_score: Mapped[int] = mapped_column(Integer)
    cliffhanger_score: Mapped[int] = mapped_column(Integer)
    dropoff_risk: Mapped[int] = mapped_column(Integer)
    analysis: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    episode: Mapped[Episode] = relationship(back_populates="versions")
    variants: Mapped[list["HookVariant"]] = relationship(back_populates="version", cascade="all, delete-orphan",
                                                         order_by="HookVariant.label")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="version", cascade="all, delete-orphan")


class HookVariant(Base):
    __tablename__ = "hook_variants"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("versions.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(4))
    genre: Mapped[str] = mapped_column(String(40))
    angle: Mapped[str] = mapped_column(String(120))
    text: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text, default="")
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    primary_metric: Mapped[str] = mapped_column(String(80), default="")
    version: Mapped[Version] = relationship(back_populates="variants")


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("versions.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    metric: Mapped[str] = mapped_column(String(80), default="3-second view rate")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    version: Mapped[Version] = relationship(back_populates="experiments")
    arms: Mapped[list["Arm"]] = relationship(back_populates="experiment", cascade="all, delete-orphan",
                                             order_by="Arm.id")


class Arm(Base):
    __tablename__ = "experiment_arms"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    variant_id: Mapped[int | None] = mapped_column(ForeignKey("hook_variants.id", ondelete="SET NULL"), nullable=True)
    label: Mapped[str] = mapped_column(String(120))
    is_control: Mapped[bool] = mapped_column(default=False)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    experiment: Mapped[Experiment] = relationship(back_populates="arms")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

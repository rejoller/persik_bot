from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import ForeignKey, String, BIGINT, TIMESTAMP, DateTime, BOOLEAN

class Base(DeclarativeBase):
    pass





class Users(Base):
    __tablename__ = 'users'
    user_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    first_name: Mapped[str] = mapped_column(String(255))
    last_name: Mapped[str] = mapped_column(String(255), nullable=True)
    username: Mapped[str] = mapped_column(String(255), nullable=True)
    joined_at: Mapped[DateTime] = mapped_column(TIMESTAMP)
    is_admin: Mapped[bool] = mapped_column(BOOLEAN)
    
    
    
class Badphrases(Base):
    __tablename__ = 'badphrases'
    phrase_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    phrase_text: Mapped[str] = mapped_column(String)
    unicoded_phrase_text: Mapped[str] = mapped_column(String, nullable=True)
    
    
    
    
class Messages(Base):
    __tablename__ = 'messages'
    message_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    message_tg_id: Mapped[int] = mapped_column(BIGINT)
    date_send: Mapped[DateTime] = mapped_column(TIMESTAMP)
    message_text: Mapped[bool] = mapped_column(String, nullable=True)
    




    
    
    
    
    
    
    
    
    
    
    
    

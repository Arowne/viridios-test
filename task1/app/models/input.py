from datetime import date
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from .project import Project


class Input(BaseModel):
    start_date: date
    end_date: date
    scenarios: List[Project]

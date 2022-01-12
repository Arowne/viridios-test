from fastapi import FastAPI
from pydantic import BaseModel
    
class ProjectData(BaseModel):
    index: str

class Project(BaseModel):
    project: ProjectData
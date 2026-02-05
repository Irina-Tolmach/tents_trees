from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from src.grid.grid import GridOptim
from src.algorithms.ilp_solver import ilp_solver

router = APIRouter()

# ======= Входная модель =======
class SizeRequest(BaseModel):
    rows: int
    cols: int
    difficulty: str

# ======= Выходная модель =======
class GridResponse(BaseModel):
    grid: List[List[int]]
    row: List[int]
    col: List[int]

class SolveRequest(BaseModel):
    grid: List[List[int]]
    row: List[int]
    col: List[int]

class SolveResponse(BaseModel):
    grid: List[List[int]]


# ======= FastAPI endpoint =======
@router.post("/generate", response_model=GridResponse)
async def generate_grid(request: SizeRequest):
    grid_obj = GridOptim(request.rows, request.cols, request.difficulty)
    return GridResponse(
        grid=grid_obj.grid.tolist(),
        row=grid_obj.row_constraints,
        col=grid_obj.col_constraints
    )

@router.post("/solve", response_model=SolveResponse)
async def solve_grid(request: SolveRequest):
    solver = ilp_solver(request.grid, request.row, request.col)
    return SolveResponse(
        grid=solver
    )

from typing import TypedDict


class TAResult(TypedDict, total=False):
    name: str
    seconds: int
    area: float
    solidity: float
    max_width: float
    max_height: float
    mean_curvature: float

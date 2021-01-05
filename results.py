from typing import TypedDict


class TAResult(TypedDict, total=False):
    id: str
    area: float
    solidity: float
    max_width: float
    max_height: float
    mean_curvature: float

from dataclasses import dataclass

@dataclass(frozen = True)
class Tile:
    tile_id: int
    cx: float
    cy: float
    color: str

@dataclass
class Event:
    MOVE = "tile_move"
    ADD = "tile_add"
    REMOVE = "tile_remove"
    etype: str
    session_id: str
    timestamp: float
    tile: Tile
    init_x: float
    init_y: float
    canvas_state: list[Tile]

@dataclass
class Session:
    num_steps: int
    goal: str
    session_id: str
    start_time: float
    canvas_width: int
    canvas_height: int
    tile_size: int
    available_colors: list[str]
    events: list[Event]
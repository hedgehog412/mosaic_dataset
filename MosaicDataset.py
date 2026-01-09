import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from DataClasses import Tile, Event, Session
import numpy as np

class MosaicDataset(Dataset):
    def __init__(self, log_dir: str|Path, \
                include_intermediate=True, copy_num=5, \
                global_seed=12, aug_img_p=0.5, \
                jitter_px=2.0, dropout_p=0.02,\
                rot_deg_max=4.0, scale_p=0.05,\
                chip_p=0.10, chip_max_frac=0.20,\
                edge_dropout_p=0.01):
        self.log_dir = Path(log_dir)
        self.sessions = [self.load_session(path) for path in self.log_dir.glob("*.jsonl")]
        self.total_sessions = len(self.sessions)
        self.total_steps = 0
        for session in self.sessions:
            self.total_steps += session.num_steps

        self.include_intermediate = include_intermediate
        self.copy_num = copy_num

        self.aug_img_p = aug_img_p
        self.jitter_px = jitter_px
        self.dropout_p = dropout_p
        self.rot_deg_max = rot_deg_max
        self.scale_p = scale_p
        self.chip_p = chip_p
        self.chip_max_frac = chip_max_frac
        self.edge_dropout_p = edge_dropout_p

        # Converting dataset idx to session, step idx if intermediate steps included
        self.base_idx = []
        if self.include_intermediate:
            for si, s in enumerate(self.sessions):
                for ei in range(s.num_steps):
                    self.base_idx.append((si, ei))

        self.global_seed = global_seed
        self.epoch = 0

    # NEEDS TO BE CALLED FOR EACH EPOCH FOR EPOCH DEPENDENT RANDOMNESS
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        if self.include_intermediate:
            return self.total_steps * self.copy_num
        else:
            return self.total_sessions * self.copy_num

    def __getitem__(self, idx):
        # For getting session/event
        base_i = idx // self.copy_num

        if self.include_intermediate:
            (si, ei) = self.base_idx[base_i]
        else:
            (si, ei) = base_i, self.sessions[base_i].num_steps - 1
        
        session = self.sessions[si]
        event = session.events[ei]

        # For getting augmentation copy
        aug_i = idx % self.copy_num
        seed = (self.global_seed + 100000 * self.epoch + 1000 * si + 10 * ei + aug_i) % (2 ** 32 - 1)
        rng = np.random.default_rng(seed = seed)
        aug_img = False
        if aug_i <= self.copy_num * self.aug_img_p:
            aug_img = True
        img = self.render_tiles(session.events[ei].canvas_state, session.canvas_width, session.canvas_height, session.tile_size, rng, aug_img=aug_img)
        return img

    def render_tiles(self, tiles: list[Tile], canvas_w: int, canvas_h: int, tile_size: int, rng: np.random.Generator, out_size=(128, 128), aug_img=False) -> torch.Tensor:
        img = np.ones((canvas_h, canvas_w, 1), dtype=np.float32)
        for t in tiles:
            dx = rng.uniform(-self.jitter_px, self.jitter_px)
            dy = rng.uniform(-self.jitter_px, self.jitter_px)
            aug_tile = Tile(t.tile_id, t.cx + dx, t.cy + dy, t.color)
            tile_mask = self.render_tile_mask(aug_tile, tile_size, rng, aug_img=aug_img)
            x_start = int(round(aug_tile.cx - tile_size / 2))
            y_start = int(round(aug_tile.cy - tile_size / 2))
            x_end = x_start + tile_size
            y_end = y_start + tile_size
            if x_start < 0 or y_start < 0 or x_end > canvas_w or y_end > canvas_h:
                continue
            img[y_start:y_end, x_start:x_end, :] *= tile_mask
        return torch.from_numpy(img)

    def render_tile_mask(self, tile: Tile, tile_size: int, rng: np.random.Generator) -> np.ndarray:
        if self.dropout_p > 0.0 and rng.uniform(0.0, 1.0) < self.dropout_p:
            return np.zeros((tile_size, tile_size, 1), dtype=np.float32)

        base = np.ones((tile_size, tile_size, 1), dtype=np.float32)
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:tile_size, :tile_size]

        if self.chip_p > 0.0 and rng.uniform(0.0, 1.0) < self.chip_p:
            frac = rng.uniform(0.0, self.chip_max_frac)
            chip_w = int(tile_size * frac)
            chip_h = int(tile_size * frac)
            # Randomly select one of the four corners
            corner = rng.integers(0, 4)
            if corner == 0:  # Top-left triangle
                mask = (x_coords < chip_w) & (y_coords < chip_h) & (x_coords + y_coords <= chip_w)
            elif corner == 1:  # Top-right triangle
                mask = (x_coords >= tile_size - chip_w) & (y_coords < chip_h) & ((tile_size - x_coords) + y_coords <= chip_w)
            elif corner == 2:  # Bottom-left triangle
                mask = (x_coords < chip_w) & (y_coords >= tile_size - chip_h) & (x_coords + (tile_size - y_coords) <= chip_w)
            else:  # Bottom-right triangle
                mask = (x_coords >= tile_size - chip_w) & (y_coords >= tile_size - chip_h) & ((tile_size - x_coords) + (tile_size - y_coords) <= chip_w)
            
            base[mask, :] = 0

        if self.edge_dropout_p > 0.0:
            edge_mask = np.zeros((tile_size, tile_size), dtype=bool)
            edge_width = int(tile_size * 0.1)
            edge_mask[:edge_width, :] = True  # Top edge
            edge_mask[-edge_width:, :] = True  # Bottom edge
            edge_mask[:, :edge_width] = True  # Left edge
            edge_mask[:, -edge_width:] = True  # Right edge

            dropout_mask = (rng.uniform(0.0, 1.0, size=(tile_size, tile_size)) < self.edge_dropout_p) & edge_mask
            base[dropout_mask, :] = 0

        if self.rot_deg_max > 0.0 or (self.scale_p > 0.0 and self.scale_p < 1.0):
            cx = tile_size / 2
            cy = tile_size / 2

            scale = 1.0 + rng.uniform(-self.scale_p, self.scale_p)
            sx = (x_coords - cx) * scale
            sy = (y_coords - cy) * scale

            angle = np.deg2rad(rng.uniform(-self.rot_deg_max, self.rot_deg_max))
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rx = cos_a * sx + sin_a * sy + cx
            ry = -sin_a * sx + cos_a * sy + cy

            xi = np.clip(np.round(rx).astype(int), 0, tile_size - 1)
            yi = np.clip(np.round(ry).astype(int), 0, tile_size - 1)
            mask = (xi >= 0) & (xi < tile_size) & (yi >= 0) & (yi < tile_size)
            base[mask, :] = base[yi[mask], xi[mask], :]
        return base

    def load_session(self, path: str|Path) -> Session:
        path = Path(path)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(lines) == 0 or lines[0].get("type") != "metadata":
            raise ValueError(f"Invalid session file: {path}")

        metadata = lines[0]
        goal = metadata.get("goal")
        session_id = metadata.get("session_id")
        start_time = float(metadata.get("start_time"))
        canvas_width = int(metadata.get("canvas_width"))
        canvas_height = int(metadata.get("canvas_height"))
        tile_size = int(metadata.get("tile_size"))
        available_colors = list(metadata.get("available_colors", []))

        session = Session(num_steps=0, goal=goal, session_id=session_id, \
                        start_time=start_time, canvas_width=canvas_width, canvas_height=canvas_height, \
                        tile_size=tile_size, available_colors=available_colors, events=[])

        active_tiles = []
        for line in lines[1:]:
            etype = line.get("type")
            session_id = line.get("session_id")
            timestamp = float(line.get("timestamp"))
            tile_id = int(line.get("tile_id"))
            if etype == Event.MOVE:
                cx = float(line.get("final_x"))
                cy = float(line.get("final_y"))
                init_x = float(line.get("init_x"))
                init_y = float(line.get("init_y"))
            else:
                cx = float(line.get("center_x"))
                cy = float(line.get("center_y"))
                init_x = cx
                init_y = cy
            color = line.get("color")
            tile = Tile(tile_id, cx, cy, color)

            
            if session_id != session.session_id:
                raise ValueError(f"Invalid session id at file: {path}\n Expected: {session.session_id}, Found: {session_id}")

            if etype == Event.ADD:
                active_tiles.append(tile)
            elif etype == Event.MOVE:
                changed = False
                for at in active_tiles:
                    if at.tile_id == tile.tile_id:
                        active_tiles.remove(at)
                        active_tiles.append(tile)
                        changed = True
                if not changed:
                    raise ValueError(f"Invalid session file: {path} for moving tile {tile}")
            elif etype == Event.REMOVE:
                if active_tiles.count(tile) == 0:
                    raise ValueError(f"Invalid session file: {path} for removing tile {tile}")
                else:
                    active_tiles.remove(tile)
            else:
                raise ValueError(f"Invalid event type at file: {path}")

            event = Event(etype, session_id, timestamp, tile, init_x, init_y, active_tiles)
            session.events.append(event)
        return session
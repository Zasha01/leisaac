from pxr import Usd, UsdGeom, Gf, UsdPhysics
import omni.usd, random, json
from collections import defaultdict
import chess, chess.pgn
import io, os
import torch
import omni.physics.tensors as physx_tensors

import re


# Board extents in chess_001 LOCAL frame
X_MIN, X_MAX = -0.164,  0.164
Y_MIN, Y_MAX = -0.164,  0.164
STEP_X = (X_MAX - X_MIN) / 7.0
STEP_Y = (Y_MAX - Y_MIN) / 7.0
Z_PLANE = -0.00832

def square_center_local(square: str) -> Gf.Vec3f:
    f = ord(square[0].lower()) - ord('a')   # 0..7
    r = int(square[1]) - 1                  # 0..7
    f = 7 - f
    r = 7 - r
    x = X_MIN + f * STEP_X
    y = Y_MIN + r * STEP_Y
    return Gf.Vec3f(x, y, Z_PLANE)


def _infer_type_from_name(n: str) -> str:
    n = n.lower()
    if "pawn" in n:   return "P"
    if "knight" in n: return "N"
    if "bishop" in n: return "B"
    if "rook" in n:   return "R"
    if "queen" in n:  return "Q"
    if "king" in n:   return "K"
    return "?"

def _color_from_start_sq(sq: str) -> str:
    r = int(sq[1])
    return "white" if r in (1,2) else "black"  # fits your mapping

def _get_envs(stage):
    root = stage.GetPrimAtPath("/World/envs")
    return [] if not root or not root.IsValid() else list(root.GetChildren())

def _get_chess_root(stage, env_ns: str):
    return stage.GetPrimAtPath(f"{env_ns}/Scene/chess/chess/chess_001")

def build_piece_catalog_from_mapping() -> dict:
    stage = omni.usd.get_context().get_stage()
    out = {"envs": {}}
    for env in _get_envs(stage):
        env_ns = env.GetPath().pathString
        chess_root = _get_chess_root(stage, env_ns)
        if not chess_root or not chess_root.IsValid():
            continue

        by_name = {p.GetName(): p for p in chess_root.GetChildren()}

        pieces = []
        for prim_name, start_sq in PRIM_TO_START_SQUARE.items():
            prim = by_name.get(prim_name)
            if not prim or not prim.IsValid():
                print(f"[chess] Missing prim '{prim_name}' under {chess_root.GetPath()}")
                continue
            typ = _infer_type_from_name(prim_name)
            color = _color_from_start_sq(start_sq)
            rb_path = _find_rb_path_for_piece(stage, prim)  # <-- NEW
            if rb_path is None:
                print(f"[chess] No RigidBodyAPI under '{prim_name}' ({prim.GetPath()}). Will fallback to USD xform for this piece.")
            pieces.append({
                "name": prim_name,
                "path": prim.GetPath().pathString,
                "rb_path": rb_path,             # <-- NEW
                "type": typ,                    # 'P','N','B','R','Q','K'
                "color": color,                 # 'white' or 'black'
            })
        out["envs"][env_ns] = {"chess_root": chess_root.GetPath().pathString, "pieces": pieces}
    return out

def _chess_local_to_world(stage, chess_root_prim, x, y, z):
    time = Usd.TimeCode.Default()
    xcache = UsdGeom.XformCache(time)
    M_c2w = xcache.GetLocalToWorldTransform(chess_root_prim)
    return M_c2w.Transform(Gf.Vec3d(x, y, z))

def _set_transforms_via_tensors(chess_root_path: str,
                                rb_path_to_local_xy: dict[str, tuple[float, float]],
                                keep_current_z: bool = True,
                                z_plane: float = Z_PLANE):
    """Move rigid pieces using Direct GPU API. Keys must be *rigid body* prim paths."""
    stage = omni.usd.get_context().get_stage()
    chess_root = stage.GetPrimAtPath(chess_root_path)
    if not chess_root or not chess_root.IsValid():
        print(f"[chess] chess root not found: {chess_root_path}")
        return

    if not rb_path_to_local_xy:
        print("[chess] No RB paths provided for tensors update.")
        return

    # Create a view from the exact RB paths we intend to move
    sim_view = physx_tensors.create_simulation_view("torch")
    rbv = sim_view.create_rigid_body_view(list(rb_path_to_local_xy.keys()))

    if rbv.count == 0:
        print("[chess] RigidBodyView contains 0 bodies; nothing to move.")
        return

    # Map path -> index in the view
    paths = list(rbv.prim_paths)  # attribute
    idx_by_path = {p: i for i, p in enumerate(paths)}

    T = rbv.get_transforms().reshape(rbv.count, 7)  # (x,y,z,qx,qy,qz,qw)

    updates, indices = [], []
    for rb_path, (lx, ly) in rb_path_to_local_xy.items():
        i = idx_by_path.get(rb_path)
        if i is None:
            # Not in the view — skip gracefully
            continue
        z_cur = T[i, 2].item() if keep_current_z else float(z_plane)
        xw, yw, zw = _chess_local_to_world(stage, chess_root, float(lx), float(ly), z_cur)
        qx, qy, qz, qw = T[i, 3:].tolist()
        updates.append([xw, yw, zw, qx, qy, qz, qw])
        indices.append(i)

    if not indices:
        print("[chess] None of the RB paths matched the view.")
        return

    idx = torch.tensor(indices, device=T.device, dtype=torch.int32)
    data = torch.tensor(updates, device=T.device, dtype=T.dtype)
    rbv.set_transforms(data, idx)
    rbv.set_velocities(torch.zeros((len(indices), 6), device=T.device, dtype=T.dtype), idx)
    print(f"[chess] Updated {len(indices)} rigid piece transforms via tensors.")



def _set_piece_xy_in_chess_local(stage, piece_prim, chess_root_prim, target_xy):
    time = Usd.TimeCode.Default()
    xcache = UsdGeom.XformCache(time)
    M_world_from_chess = xcache.GetLocalToWorldTransform(chess_root_prim)
    M_chess_from_world = Gf.Matrix4d(M_world_from_chess).GetInverse()

    bbox = UsdGeom.BBoxCache(time, includedPurposes=[UsdGeom.Tokens.default_], useExtentsHint=True)
    aabb = bbox.ComputeWorldBound(piece_prim).ComputeAlignedBox()
    c_world = (aabb.GetMin() + aabb.GetMax()) * 0.5
    c_local = M_chess_from_world.Transform(c_world)

    new_local = Gf.Vec3f(target_xy[0], target_xy[1], -0.00832)  # keep current local Z

    xf = UsdGeom.Xformable(piece_prim)
    ops = xf.GetOrderedXformOps()
    t_op = None
    for op in ops:
        if op.GetOpName().startswith("xformOp:translate"):
            t_op = op; break
    if t_op is None:
        t_op = xf.AddTranslateOp()
    t_op.Set(new_local)


def _parse_fen_board(fen: str) -> dict:
    """Return {square: fenChar} for board part of FEN."""
    board = fen.split()[0]
    squares = {}
    r = 7
    f = 0
    for ch in board:
        if ch == "/":
            r -= 1; f = 0
        elif ch.isdigit():
            f += int(ch)
        else:
            sq = chr(ord('a') + f) + str(r + 1)
            squares[sq] = ch
            f += 1
    return squares

def apply_fen_all_envs(catalog: dict, fen: str, park_offboard=False):
    stage = omni.usd.get_context().get_stage()
    target = _parse_fen_board(fen)

    for env_ns, data in catalog["envs"].items():
        chess_root_path = data["chess_root"]
        chess_root = stage.GetPrimAtPath(chess_root_path)
        if not chess_root or not chess_root.IsValid():
            continue

        # path -> rb_path lookup
        path_to_rb = {e["path"]: e.get("rb_path") for e in data["pieces"]}

        # bucket by code
        pool = defaultdict(list)
        for e in data["pieces"]:
            prim = stage.GetPrimAtPath(e["path"])
            if not prim or not prim.IsValid():
                continue
            code = e["type"] if e["color"] == "white" else e["type"].lower()
            pool[code].append(prim)

        rb_updates: dict[str, tuple[float, float]] = {}
        fallback_updates: list[tuple[Usd.Prim, tuple[float, float]]] = []

        used = set()

        # place required pieces
        for sq, code in target.items():
            if not pool.get(code):
                print(f"[chess] ({env_ns}) No piece available for '{code}' at {sq}")
                continue
            prim = pool[code].pop()
            used.add(prim)
            p = square_center_local(sq)
            rb_path = path_to_rb.get(prim.GetPath().pathString)
            if rb_path:
                rb_updates[rb_path] = (float(p[0]), float(p[1]))
            else:
                fallback_updates.append((prim, (float(p[0]), float(p[1]))))

        # park remaining pieces offboard
        if park_offboard:
            off_x = X_MAX + 0.05
            y0 = Y_MIN
            dy = 0.03
            i = 0
            for lst in pool.values():
                for prim in lst:
                    if prim in used: continue
                    y = y0 + (i % 8) * dy
                    rb_path = path_to_rb.get(prim.GetPath().pathString)
                    if rb_path:
                        rb_updates[rb_path] = (float(off_x), float(y))
                    else:
                        fallback_updates.append((prim, (float(off_x), float(y))))
                    i += 1

        # 1) move all RBs via tensors
        if rb_updates:
            _set_transforms_via_tensors(chess_root_path, rb_updates, keep_current_z=False, z_plane=0.025)

        # 2) fallback: USD xform for non-RB pieces (rare)
        for prim, (lx, ly) in fallback_updates:
            _set_piece_xy_in_chess_local(stage, prim, chess_root, (lx, ly))

        print(f"[chess] Applied FEN in {env_ns}")




def random_fen_from_db(pgn_path: str | None = None,
                       fen_list_path: str | None = None,
                       min_plies: int = 10,
                       max_plies: int = 80) -> str:
    # Try PGN
    if pgn_path:
        try:
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
                games_offsets = []
                while True:
                    off = f.tell()
                    if not chess.pgn.read_headers(f): break
                    games_offsets.append(off)
                    chess.pgn.skip_game(f)
                if not games_offsets:
                    raise RuntimeError("No games in PGN.")
                # pick game, then ply
                off = random.choice(games_offsets)
                f.seek(off)
                game = chess.pgn.read_game(f)
                b = game.board()
                plies = random.randint(min_plies, max_plies)
                for _ in range(plies):
                    if b.is_game_over(): break
                    moves = list(b.legal_moves)
                    if not moves: break
                    b.push(random.choice(moves))
                return b.fen()
        except Exception as e:
            print(f"[chess] PGN load failed ({e}); falling back.")

    # Try FEN list
    if fen_list_path:
        try:
            with open(fen_list_path, "r") as f:
                fens = [line.strip() for line in f if line.strip()]
            if fens:
                return random.choice(fens)
        except Exception as e:
            print(f"[chess] FEN list load failed ({e}); falling back.")

    # Fallback small set
    FALLBACK_FENS = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r4rk1/ppp2ppp/2n1bn2/3q4/3P4/2P2N2/PPQ2PPP/R1B2RK1 w - - 6 12",
        "r1bq1rk1/pp1n1pbp/2p1p1p1/3pP3/3P1P2/2NBB3/PPQ3PP/R3K2R w KQ - 0 12",
    ]
    return random.choice(FALLBACK_FENS)


def init_random_position_from_db(pgn_path: str | None = None,
                                 fen_list_path: str | None = None,
                                 save_catalog_path: str | None = None):
    catalog = build_piece_catalog_from_mapping()
    if save_catalog_path:
        with open(save_catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"[chess] Catalog saved to {save_catalog_path}")

    fen = random_fen_from_db(pgn_path=pgn_path, fen_list_path=fen_list_path)
    apply_fen_all_envs(catalog, fen)
    print(f"[chess] Random FEN: {fen}")
    return fen

def _has_rigid_body(stage, prim) -> bool:
    """Return True if the RigidBody API is applied on prim."""
    api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    # If the API is applied, its authored attributes are valid on this prim.
    attr = api.GetRigidBodyEnabledAttr()
    return bool(attr and attr.IsValid())

def _find_rb_path_for_piece(stage, piece_prim):
    """
    Return the prim path (string) of the first prim (self or descendant)
    that has UsdPhysics.RigidBodyAPI applied. Else return None.
    """
    # self
    if _has_rigid_body(stage, piece_prim):
        return piece_prim.GetPath().pathString
    # search descendants (your meshes like .../Pawn_001/Pawn_001)
    stack = list(piece_prim.GetChildren())
    while stack:
        p = stack.pop()
        if _has_rigid_body(stage, p):
            return p.GetPath().pathString
        stack.extend(p.GetChildren())
    return None

# --- YOUR VERIFIED PRIM → START SQUARES (in chess_001 local frame) ---
PRIM_TO_START_SQUARE = {
    "Bishop_003": "f1",
    "Knight":     "b8",
    "Knight_001": "g8",
    "Rook_003":   "h1",
    "Queen_001":  "d1",
    "Rook_002":   "a1",
    "Rook_001":   "h8",
    "Rook":       "a8",
    "Pawn_015":   "h2",
    "Pawn_008":   "a2",
    "Pawn_010":   "c2",
    "Queen":      "d8",
    "Pawn_012":   "e2",
    "Pawn_009":   "b2",
    "Pawn_004":   "e7",
    "Pawn_005":   "f7",
    "Pawn_014":   "g2",
    "Pawn_013":   "f2",
    "Pawn_011":   "d2",
    "Pawn_007":   "h7",
    "Pawn_006":   "g7",
    "King":       "e8",
    "Bishop_001": "f8",
    "King_001":   "e1",
    "Knight_002": "b1",
    "Bishop_002": "c1",
    "Knight_003": "g1",
    "Bishop":     "c8",
    "Pawn":       "a7",
    "Pawn_002":   "c7",
    "Pawn_003":   "d7",
    "Pawn_001":   "b7",
}


# --- inverse mapping: local (x,y) -> algebraic square (respects FLIP_* like your _square_center)
def _square_from_local_xy(x: float, y: float) -> str:
    fx = round((x - X_MIN) / STEP_X)
    fy = round((y - Y_MIN) / STEP_Y)
    fx = int(max(0, min(7, fx)))
    fy = int(max(0, min(7, fy)))
    # invert the flips done in _square_center
    if True:
        fx = 7 - fx
    if True:
        fy = 7 - fy
    return chr(ord('a') + fx) + str(fy + 1)

# --- tensors-based board reader (uses your catalog with rb_path)
def read_board_from_sim_gpu(catalog: dict) -> chess.Board:
    stage = omni.usd.get_context().get_stage()
    env_items = list(catalog["envs"].items())
    if not env_items:
        raise RuntimeError("No envs in catalog.")
    env_ns, data = env_items[0]  # first env
    chess_root = stage.GetPrimAtPath(data["chess_root"])
    if not chess_root or not chess_root.IsValid():
        raise RuntimeError(f"Invalid chess root: {data['chess_root']}")

    # world<->local transforms for chess_001
    time = Usd.TimeCode.Default()
    xcache = UsdGeom.XformCache(time)
    M_c2w = xcache.GetLocalToWorldTransform(chess_root)
    M_w2c = Gf.Matrix4d(M_c2w).GetInverse()

    rb_paths = [e["rb_path"] for e in data["pieces"] if e.get("rb_path")]
    names    = [e["name"]    for e in data["pieces"] if e.get("rb_path")]
    types    = [e["type"]    for e in data["pieces"] if e.get("rb_path")]
    colors   = [e["color"]   for e in data["pieces"] if e.get("rb_path")]

    if not rb_paths:
        raise RuntimeError("Catalog has no rb_path entries. Rebuild it with build_piece_catalog_from_mapping().")

    sim_view = physx_tensors.create_simulation_view("torch")
    rbv = sim_view.create_rigid_body_view(rb_paths)
    if rbv.count == 0:
        raise RuntimeError("RigidBodyView is empty.")

    T = rbv.get_transforms().reshape(rbv.count, 7)  # (x,y,z,qx,qy,qz,qw)

    board = chess.Board.empty()
    for i, rb_path in enumerate(rbv.prim_paths):
        px, py, pz = float(T[i,0]), float(T[i,1]), float(T[i,2])
        p_local = M_w2c.Transform(Gf.Vec3d(px, py, pz))
        sq = _square_from_local_xy(p_local[0], p_local[1])
        t  = types[i]   # 'P','N','B','R','Q','K'
        c  = colors[i]  # 'white' or 'black'
        piece = chess.Piece.from_symbol(t if c == "white" else t.lower())
        board.set_piece_at(chess.parse_square(sq), piece)
    # Rough guess at side to move: keep 'w' unless you track it elsewhere
    board.turn = chess.WHITE
    return board

# --- tiny evaluation (material + PST + a pinch of mobility)
PST = {
    chess.PAWN:   [ 0,  5,  5,  0,  5, 10, 20,  0 ] * 8,
    chess.KNIGHT: [ -5,  0,  5, 10, 10,  5,  0, -5 ] * 8,
    chess.BISHOP: [ 0, 10, 10, 15, 15, 10, 10,  0 ] * 8,
    chess.ROOK:   [ 5, 10, 10, 15, 15, 10, 10,  5 ] * 8,
    chess.QUEEN:  [ 0,  0,  5, 10, 10,  5,  0,  0 ] * 8,
    chess.KING:   [ 0,  0,-10,-20,-20,-10,  0,  0 ] * 8,
}
VALUES = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}

def evaluate(board: chess.Board) -> int:
    score = 0
    for sq, piece in board.piece_map().items():
        val = VALUES[piece.piece_type]
        pst = PST.get(piece.piece_type, [0]*64)
        idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
        v = val + pst[idx]
        score += v if piece.color == chess.WHITE else -v
    # mobility
    score += 2 * (len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves)))
    return score

def search(board: chess.Board, depth: int = 2, alpha: int = -10**9, beta: int = 10**9) -> tuple[int, chess.Move | None]:
    if depth == 0 or board.is_game_over():
        return evaluate(board), None
    best_move = None
    if board.turn == chess.WHITE:
        best = -10**9
        for mv in board.legal_moves:
            board.push(mv)
            val, _ = search(board, depth-1, alpha, beta)
            board.pop()
            if val > best:
                best, best_move = val, mv
            alpha = max(alpha, val)
            if beta <= alpha: break
        return best, best_move
    else:
        best = 10**9
        for mv in board.legal_moves:
            board.push(mv)
            val, _ = search(board, depth-1, alpha, beta)
            board.pop()
            if val < best:
                best, best_move = val, mv
            beta = min(beta, val)
            if beta <= alpha: break
        return best, best_move
def _square_center(square: str) -> Gf.Vec3f:
    m = re.fullmatch(r'([a-hA-H])([1-8])', square.strip())
    if not m:
        raise ValueError(f"Invalid square '{square}'. Use like 'e5'.")
    file_idx = ord(m.group(1).lower()) - ord('a')   # 0..7
    rank_idx = int(m.group(2)) - 1                  # 0..7

    if True:
        file_idx = 7 - file_idx
    if True:
        rank_idx = 7 - rank_idx

    x = X_MIN + file_idx * STEP_X
    y = Y_MIN + rank_idx * STEP_Y
    return Gf.Vec3f(x, y, Z_PLANE)

# --- highlighting helpers (reuse your code path; add color/prefix)
def _highlight_square_colored(square: str, color_rgb=(1.0, 0.0, 0.0), prefix="Highlight"):
    stage = omni.usd.get_context().get_stage()
    envs_root = stage.GetPrimAtPath("/World/envs")
    if not envs_root or not envs_root.IsValid():
        print("[highlight] /World/envs not ready.")
        return
    pos = _square_center(square)
    for env_prim in envs_root.GetChildren():
        env_ns = env_prim.GetPath().pathString
        chess_root = stage.GetPrimAtPath(f"{env_ns}/Scene/chess/chess/chess_001")
        if not chess_root or not chess_root.IsValid():
            continue
        cube_path = f"{chess_root.GetPath()}/{prefix}_{square.lower()}"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        xform = UsdGeom.Xformable(cube)
        if xform.GetOrderedXformOps():
            xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(pos)
        xform.AddScaleOp().Set(Gf.Vec3f(0.022, 0.022, 0.0002))
        cube.CreateDisplayColorAttr([color_rgb])

def highlight_move(from_sq: str, to_sq: str):
    _highlight_square_colored(from_sq, color_rgb=(0.0, 1.0, 0.0), prefix="HighlightFrom")  # green
    _highlight_square_colored(to_sq,   color_rgb=(1.0, 0.0, 0.0), prefix="HighlightTo")    # red

# --- top-level: compute & highlight best move
def compute_and_highlight_best_move(
    depth: int = 2,
    pgn_path: str | None = None,
    fen_list_path: str | None = None,
    save_catalog_path: str | None = None,
    step_env: "callable | None" = None,   # e.g. lambda: env.sim.step(render=True)
):
    """
    Randomize a chess position, compute an approx-best move, and highlight it.

    Args:
        depth: search depth for the tiny engine.
        pgn_path: optional PGN file to sample random positions from.
        fen_list_path: optional text file with FENs (one per line).
        save_catalog_path: optional JSON path to dump the piece catalog.
        step_env: optional callable to advance physics once (e.g., lambda: env.sim.step(render=True)).
                  With Direct GPU API, stepping once helps the viewport reflect the new transforms immediately.
    Returns:
        (fen, move, score) where move is a chess.Move or None.
    """
    # 0) clear old highlights
    clear_highlight_squares()
    
    # 1) build / persist catalog (includes rb_path per piece)
    catalog = build_piece_catalog_from_mapping()
    if save_catalog_path:
        with open(save_catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"[chess] Catalog saved to {save_catalog_path}")

    # 2) pick a random FEN and apply to sim
    fen = random_fen_from_db(pgn_path=pgn_path, fen_list_path=fen_list_path)
    apply_fen_all_envs(catalog, fen)

    # 3) (optional) tick physics so the viewport updates immediately
    if step_env is not None:
        try:
            step_env()  # one small step is enough; caller can pass lambda: env.sim.step(render=True)
        except Exception as e:
            print(f"[chess] step_env callable raised: {e}")

    # 4) read current board from tensors & 5) search best move
    board = read_board_from_sim_gpu(catalog)
    score, mv = search(board, depth=depth)

    if mv is None:
        print("[engine] No legal move found.")
        return fen, None, score

    from_sq = chess.square_name(mv.from_square)
    to_sq   = chess.square_name(mv.to_square)

    # 6) highlight best move
    highlight_move(from_sq, to_sq)

    print(f"[engine] depth={depth} best ~ {board.san(mv)}  (from {from_sq} to {to_sq}, eval {score})")
    return fen, mv, score
# ========= END =========


def clear_highlight_squares(prefixes=("Highlight", "HighlightFrom", "HighlightTo")):
    """Delete any existing highlight cubes in every env's chess_001 subtree."""
    stage = omni.usd.get_context().get_stage()
    envs_root = stage.GetPrimAtPath("/World/envs")
    if not envs_root or not envs_root.IsValid():
        return
    removed = 0
    for env_prim in envs_root.GetChildren():
        env_ns = env_prim.GetPath().pathString
        chess_root = stage.GetPrimAtPath(f"{env_ns}/Scene/chess/chess/chess_001")
        if not chess_root or not chess_root.IsValid():
            continue
        # collect first (don’t mutate while iterating)
        to_delete = []
        for child in chess_root.GetChildren():
            name = child.GetName()
            if any(name.lower().startswith(pfx.lower()) for pfx in prefixes):
                to_delete.append(child.GetPath())
        for p in to_delete:
            stage.RemovePrim(p)
            removed += 1
    if removed:
        print(f"[highlight] Cleared {removed} old highlight prim(s).")
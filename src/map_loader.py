from typing import List, Tuple

Pos = Tuple[int, int]


def load_map_txt(path: str):
    """
    读取字符地图
    返回:
      grid: List[str]
      walls: set[(x,y)]
      npc_spawns: List[(x,y)]
      player_spawn: (x,y)
      h, w
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    h = len(lines)
    w = len(lines[0]) if h > 0 else 0

    for row in lines:
        if len(row) != w:
            raise ValueError("地图每行长度必须一致")

    walls = set()
    npc_spawns: List[Pos] = []
    player_spawn = None

    for i in range(h):
        for j in range(w):
            c = lines[i][j]
            if c == "#":
                walls.add((i, j))
            elif c == "S":
                npc_spawns.append((i, j))
            elif c == "P":
                player_spawn = (i, j)
            elif c == ".":
                pass
            else:
                raise ValueError(f"非法地图字符: {c} at ({i},{j})")

    if player_spawn is None:
        raise ValueError("地图必须包含一个 P 作为玩家出生点")
    if len(npc_spawns) == 0:
        raise ValueError("地图必须至少包含一个 S 作为NPC出生点")

    return lines, walls, npc_spawns, player_spawn, h, w
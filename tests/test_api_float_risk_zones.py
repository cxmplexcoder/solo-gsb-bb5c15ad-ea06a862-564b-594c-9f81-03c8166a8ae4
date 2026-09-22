"""GET /designs/{id}/float-risk-zones 的接口集成测试。

使用临时 SQLite 文件与 FastAPI TestClient，不依赖运行中的服务，也不触碰
仓库默认的 data/weave.db。直接执行：python3 tests/test_api_float_risk_zones.py
（也可被 pytest 收集）。
"""

import os
import sys
import tempfile

_TMP_DB = tempfile.NamedTemporaryFile(prefix="weave-test-", suffix=".db", delete=False)
_TMP_DB.close()
os.environ["DATABASE_PATH"] = _TMP_DB.name

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# 单列设计：tie_up=[[1],[0]]，treadling 中间一纬为 1
# 交织矩阵（1 列 5 行）：[1,1,0,1,1]
# cyclic=true 时跨边界合并成 start=3、长度 4 的经向浮长（3,4,0,1 行）
SEAM = {
    "name": "single end seam float",
    "cyclic": True,
    "max_float": 2,
    "threading": [0],
    "tie_up": [[1], [0]],
    "treadling": [0, 0, 1, 0, 0],
}

# 非循环：交织矩阵前 3 行全 1、底行全 0（3 经 4 纬）
# 经向 3 长违规 3 条（3x3 共 9 单元）+ 底纬 3 长纬浮 1 条，上下相邻合并为整幅一区
WARP_BLOCK = {
    "name": "3x3 warp block",
    "max_float": 2,
    "threading": [0, 0, 0],
    "tie_up": [[1], [0]],
    "treadling": [0, 0, 0, 1],
}

# 5x5 非循环交织矩阵：
# 1 1 0 1 0
# 1 0 1 0 1
# 1 1 0 1 0
# 0 1 0 0 1
# 1 0 0 0 0
# 违规仅两条：列 0 行 0..2 的 3 长经浮；行 4 列 1..4 的 4 长纬浮。
# 列 0 段止于 (2,0)，纬浮段起于 (4,1)，二者既不共边也不共点 -> 两个区域
TWO_ZONES = {
    "name": "two distant runs",
    "max_float": 2,
    "threading": [0, 1, 2, 3, 4],
    "tie_up": [
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ],
    "treadling": [0, 1, 0, 2, 3],
}

# 2/2 斜纹 8x8 循环：max_float=2，无任何违规
TWILL = {
    "name": "2/2 twill, 8x8 repeat",
    "cyclic": True,
    "max_float": 2,
    "threading": [0, 1, 2, 3, 0, 1, 2, 3],
    "tie_up": [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ],
    "treadling": [0, 1, 2, 3, 0, 1, 2, 3],
}


def test_float_risk_zones_endpoint():
    with TestClient(app) as client:
        # 设计不存在 -> 404
        resp = client.get("/designs/99999/float-risk-zones")
        check(resp.status_code == 404, "missing design returns 404")
        check(resp.json()["detail"] == "design 99999 not found", "404 detail message")

        # 无违规设计 -> 空列表、total_zones=0
        tid = client.post("/designs", json=TWILL).json()["id"]
        resp = client.get(f"/designs/{tid}/float-risk-zones")
        check(resp.status_code == 200, "clean design returns 200")
        data = resp.json()
        check(data["zones"] == [] and data["total_zones"] == 0,
              "no violations -> empty zones and zero total")
        check(data["design_id"] == tid and data["cyclic"] is True
              and data["warp_ends"] == 8 and data["weft_picks"] == 8
              and data["max_float"] == 2,
              "response echoes design id, cyclic, size and max_float")

        # 循环跨缝经向浮长：单区 4 单元，跨纬向接缝
        sid = client.post("/designs", json=SEAM).json()["id"]
        data = client.get(f"/designs/{sid}/float-risk-zones").json()
        check(data["total_zones"] == 1 and len(data["zones"]) == 1,
              "seam float aggregates to one zone")
        zone = data["zones"][0]
        check([(c["row"], c["col"]) for c in zone["cells"]]
              == [(0, 0), (1, 0), (3, 0), (4, 0)],
              "cells are the wrapped run (rows 3,4,0,1) in row-major order")
        check(zone["area"] == 4, "zone area is 4 cells")
        check(zone["max_float_length"] == 4, "max float length is 4")
        check(zone["directions"] == ["warp"], "seam float is warp-only")
        bbox = zone["bbox"]
        check(bbox["min_row"] == 0 and bbox["max_row"] == 4
              and bbox["min_col"] == 0 and bbox["max_col"] == 0
              and bbox["height"] == 4 and bbox["width"] == 1
              and bbox["crosses_weft_seam"] is True
              and bbox["crosses_warp_seam"] is False,
              "seam bbox touches both row edges, wrapped height 4")

        # 对照：同样长度 4 的非循环经浮（连续 1 段、无回绕），包围盒即普通矩形
        PLAIN = {
            "name": "single end plain float",
            "max_float": 2,
            "threading": [0],
            "tie_up": [[1], [0]],
            "treadling": [0, 0, 0, 0, 1],
        }
        nid = client.post("/designs", json=PLAIN).json()["id"]
        data = client.get(f"/designs/{nid}/float-risk-zones").json()
        zone = data["zones"][0]
        check(data["cyclic"] is False, "non-cyclic response echoes cyclic=false")
        check([(c["row"], c["col"]) for c in zone["cells"]]
              == [(0, 0), (1, 0), (2, 0), (3, 0)],
              "non-cyclic run does not wrap")
        bbox = zone["bbox"]
        check(bbox == {"min_row": 0, "max_row": 3, "min_col": 0, "max_col": 0,
                       "height": 4, "width": 1,
                       "crosses_weft_seam": False, "crosses_warp_seam": False},
              "non-cyclic bbox is a plain bounding box")

        # 非循环：三条并列经向 3 长列段与底纬一条 3 长纬浮上下相邻，
        # 全部合并成 4x3=12 单元的一区
        bid = client.post("/designs", json=WARP_BLOCK).json()["id"]
        audit = client.get(f"/designs/{bid}/audit").json()
        check(len(audit["violations"]) == 4,
              "warp block has three warp runs plus one bottom weft run")
        data = client.get(f"/designs/{bid}/float-risk-zones").json()
        check(data["total_zones"] == 1, "adjacent warp and weft runs merge into one zone")
        zone = data["zones"][0]
        check(zone["area"] == 12
              and [(c["row"], c["col"]) for c in zone["cells"]]
              == [(r, c) for r in range(4) for c in range(3)],
              "zone covers the full 4x3 matrix in row-major order")
        check(zone["max_float_length"] == 3
              and zone["directions"] == ["warp", "weft"],
              "merged zone carries both directions")
        check(zone["bbox"] == {"min_row": 0, "max_row": 3, "min_col": 0,
                              "max_col": 2, "height": 4, "width": 3,
                              "crosses_weft_seam": False,
                              "crosses_warp_seam": False},
              "merged zone bbox is the whole matrix")

        # 两个不相邻的违规区域：4 长纬浮排前、3 长经浮排后
        rid = client.post("/designs", json=TWO_ZONES).json()["id"]
        data = client.get(f"/designs/{rid}/float-risk-zones").json()
        check(data["total_zones"] == 2, "two distant runs stay as two zones")
        first, second = data["zones"]
        check(first["max_float_length"] == 4 and first["area"] == 4
              and first["directions"] == ["weft"],
              "severity order: length-4 weft zone first")
        check(second["max_float_length"] == 3 and second["area"] == 3
              and second["directions"] == ["warp"],
              "length-3 warp zone second")
        check([(c["row"], c["col"]) for c in first["cells"]]
              == [(4, 1), (4, 2), (4, 3), (4, 4)],
              "first zone is the horizontal weft run")
        check([(c["row"], c["col"]) for c in second["cells"]]
              == [(0, 0), (1, 0), (2, 0)],
              "second zone is the vertical warp run")

        # 严重度排序对整体响应成立
        keys = [(z["max_float_length"], z["area"]) for z in data["zones"]]
        check(keys == sorted(keys, key=lambda k: (-k[0], -k[1])),
              "zones sorted by max float then area descending")

        # 区域来自已存审计：与 /audit 中违规记录展开后的单元总数一致
        audit = client.get(f"/designs/{rid}/audit").json()
        total_run_cells = sum(v["length"] for v in audit["violations"])
        total_zone_cells = sum(z["area"] for z in data["zones"])
        check(total_run_cells == total_zone_cells == 7,
              "zone cells partition all violation run cells without overlap")

        # 接口只读：不改库、不新增设计，已存审计内容不变
        stored_before = client.get(f"/designs/{sid}").json()
        ids_before = {d["id"] for d in client.get("/designs").json()}
        for did in (tid, sid, nid, bid, rid):
            client.get(f"/designs/{did}/float-risk-zones")
        ids_after = {d["id"] for d in client.get("/designs").json()}
        check(ids_before == ids_after, "float-risk-zones creates no new designs")
        stored_after = client.get(f"/designs/{sid}").json()
        check(stored_after == stored_before, "stored design and audit are untouched")


if __name__ == "__main__":
    test_float_risk_zones_endpoint()
    try:
        os.unlink(_TMP_DB.name)
    except OSError:
        pass
    print("all API tests passed")

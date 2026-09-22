"""机织纹样浮长审计 HTTP API。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from . import db, weave
from .models import (
    AuditOut,
    BalanceHotspotsIn,
    BalanceHotspotsOut,
    DesignCompareIn,
    DesignCompareOut,
    DesignIn,
    DesignOut,
    DesignSummary,
    ErrorOut,
    FloatRiskZonesOut,
    LoomFitIn,
    LoomFitOut,
    RepeatPeriodOut,
    RepairPlanOut,
    ThresholdProfileOut,
    TieUpSuggestionOut,
    Violation,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时建表（幂等）
    db.init_db()
    yield


app = FastAPI(
    title="Weave Float Audit API",
    summary="机织纹样浮长审计：生成经纬交织矩阵并审计超限浮长",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/designs",
    status_code=201,
    response_model=DesignOut,
    responses={400: {"model": ErrorOut, "description": "维度不一致或索引越界，整份设计被拒绝"}},
    tags=["designs"],
)
def submit_design(design: DesignIn) -> dict:
    """提交设计：校验 → 生成交织矩阵 → 浮长审计 → 入库，返回完整结果。"""
    try:
        shafts, treadles = weave.validate_design(
            design.threading, design.tie_up, design.treadling, design.max_float
        )
        interlacing, violations = weave.audit_floats(
            design.threading,
            design.tie_up,
            design.treadling,
            design.max_float,
            design.cyclic,
        )
    except weave.DesignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    violation_dicts = [v.to_dict() for v in violations]
    design_id = db.insert_design_with_audit(
        name=design.name,
        cyclic=design.cyclic,
        max_float=design.max_float,
        shafts=shafts,
        treadles=treadles,
        threading_map=design.threading,
        tie_up=design.tie_up,
        treadling=design.treadling,
        interlacing=interlacing,
        violations=violation_dicts,
    )
    stored = db.get_design(design_id)
    assert stored is not None  # 刚写入，必然存在
    return stored


@app.get(
    "/designs",
    response_model=list[DesignSummary],
    tags=["designs"],
)
def list_designs() -> list[dict]:
    """列出全部已入库的合法设计（摘要）。"""
    return db.list_designs()


@app.get(
    "/designs/{design_id}",
    response_model=DesignOut,
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def get_design(design_id: int) -> dict:
    """按 ID 查询已入库的设计及其审计结果。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    return stored


@app.get(
    "/designs/{design_id}/audit",
    response_model=AuditOut,
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def get_audit(design_id: int) -> dict:
    """按设计 ID 单独查询审计结果。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    return stored["audit"]


@app.get(
    "/designs/{design_id}/float-risk-zones",
    response_model=FloatRiskZonesOut,
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def get_float_risk_zones(design_id: int) -> dict:
    """风险区域聚合：读取已入库的交织矩阵与原浮长违规记录（不重新审计），
    将每条超限经/纬浮长从其连续段起点展开为矩阵单元，在经纬相邻（上、下、
    左、右四邻接，对角不算）规则下合并为连通区域；设计声明循环（cyclic）时
    矩阵首尾相接，跨越末行/首行、末列/首列的单元同样连通。

    返回按严重度降序的区域列表（最大浮长、面积依次降序，平局取最小单元坐标
    升序），每区给出单元坐标、包围盒（跨接缝区域回显 crosses_*_seam 与
    回绕覆盖尺寸）、面积、最大浮长及涉及方向，并返回区域总数 total_zones。
    设计不存在返回 404；设计无超限浮长时 zones 为空列表、total_zones 为 0。
    接口只读计算，不改库。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    audit = stored["audit"]
    zones = weave.aggregate_float_risk_zones(
        audit["interlacing"], audit["violations"], stored["cyclic"]
    )
    return {
        "design_id": design_id,
        "cyclic": stored["cyclic"],
        "warp_ends": stored["warp_ends"],
        "weft_picks": stored["weft_picks"],
        "max_float": stored["max_float"],
        "zones": [z.to_dict() for z in zones],
        "total_zones": len(zones),
    }


@app.get(
    "/designs/{design_id}/repeat-period",
    response_model=RepeatPeriodOut,
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def get_repeat_period(design_id: int) -> dict:
    """最小循环单元分析：读取已入库设计并重新生成交织矩阵，在经、纬两个方向
    寻找能无缝平铺整幅矩阵的最小正周期（周期整除对应尺寸），返回循环单元。
    比较基于交织矩阵本身，而非分别检查穿综/踏综；取模回绕比较对声明循环
    （cyclic）设计的首尾边界同样成立。任一方向不存在更小周期时取整幅尺寸。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    period = weave.analyze_repeat_period(
        stored["threading"],
        stored["tie_up"],
        stored["treadling"],
        stored["max_float"],
    )
    return period.to_dict()


@app.get(
    "/designs/{design_id}/threshold-profile",
    response_model=ThresholdProfileOut,
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def get_threshold_profile(
    design_id: int,
    max_float_limit: int | None = Query(
        default=None,
        ge=1,
        le=64,
        description="逐档审计的阈值上限（1～64）；缺省取原设计的 max_float"
        "（若超过 64 则按 64 封顶）",
    ),
) -> dict:
    """阈值敏感性分析：读取已入库设计，在 1..max_float_limit 内逐档按原
    cyclic 语义重跑完整经纬浮长审计，返回每档阈值、经向/纬向违规数、总数，
    以及该设计达到零违规的最小阈值（上限内达不到则为 null）。接口只读
    计算，不修改数据库；设计不存在返回 404。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    limit = (
        max_float_limit
        if max_float_limit is not None
        else min(stored["max_float"], 64)
    )
    profile = weave.threshold_profile(
        stored["threading"],
        stored["tie_up"],
        stored["treadling"],
        stored["cyclic"],
        limit,
    )
    return {
        "design_id": design_id,
        "cyclic": stored["cyclic"],
        "max_float": stored["max_float"],
        **profile.to_dict(),
    }


@app.post(
    "/designs/{design_id}/tie-up-suggestions",
    response_model=list[TieUpSuggestionOut],
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def suggest_tie_up(
    design_id: int,
    limit: int = Query(default=50, ge=1, le=50, description="返回候选数上限"),
) -> list[dict]:
    """单步绑综修复建议：逐一翻转 tie_up 的一个 0/1 单元并按原 cyclic、
    max_float 重跑完整审计，仅返回违规总数严格减少的候选（不修改原设计）。
    结果按剩余违规数、受影响交织点数、踏板索引、综框索引升序排列。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    suggestions = weave.suggest_tie_up_flips(
        stored["threading"],
        stored["tie_up"],
        stored["treadling"],
        stored["max_float"],
        stored["cyclic"],
    )
    return [s.to_dict() for s in suggestions[:limit]]


@app.post(
    "/designs/{design_id}/repair-plans",
    response_model=list[RepairPlanOut],
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def repair_plans(    design_id: int,
    max_changes: int = Query(
        default=2, ge=1, le=3, description="最多翻转绑综单元的次数（1～3）"
    ),
    limit: int = Query(default=50, ge=1, le=50, description="返回方案数上限"),
) -> list[dict]:
    """多步绑综修复方案：在最多 max_changes 次绑综单元翻转内搜索终态违规总数
    严格少于原设计的方案。中间状态允许违规数不下降；按绑综状态去重，同一单元
    不在一条序列内重复翻转；等价终态只保留排序最优的序列。接口只读计算，
    不修改原设计；设计不存在返回 404，无改善时返回空列表 []。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    plans = weave.search_repair_plans(
        stored["threading"],
        stored["tie_up"],
        stored["treadling"],
        stored["max_float"],
        stored["cyclic"],
        max_changes=max_changes,
        limit=limit,
    )
    return [p.to_dict() for p in plans]


@app.post(
    "/designs/{design_id}/compare",
    response_model=DesignCompareOut,
    responses={
        400: {"model": ErrorOut, "description": "两份设计的交织矩阵尺寸不一致"},
        404: {"model": ErrorOut, "description": "任一设计不存在"},
    },
    tags=["designs"],
)
def compare_designs(design_id: int, request: DesignCompareIn) -> dict:
    """设计对齐比对：读取两份已入库设计并重新生成交织矩阵（尺寸必须一致，
    否则 400），枚举第二份矩阵在经、纬方向的所有循环错位，取不一致单元最少
    的对齐结果；平局按行偏移再列偏移升序。返回最佳偏移、错位数量与比例、
    错位坐标、按行/列汇总及对齐后的矩阵。接口只读计算，不改库。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    other = db.get_design(request.other_design_id)
    if other is None:
        raise HTTPException(
            status_code=404, detail=f"design {request.other_design_id} not found"
        )
    matrix_a = weave.build_interlacing(
        stored["threading"], stored["tie_up"], stored["treadling"]
    )
    matrix_b = weave.build_interlacing(
        other["threading"], other["tie_up"], other["treadling"]
    )
    try:
        result = weave.best_alignment(matrix_a, matrix_b)
    except weave.DesignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "design_id": design_id,
        "other_design_id": request.other_design_id,
        "warp_ends": stored["warp_ends"],
        "weft_picks": stored["weft_picks"],
        **result.to_dict(),
    }


@app.post(
    "/designs/{design_id}/balance-hotspots",
    response_model=BalanceHotspotsOut,
    responses={
        400: {"model": ErrorOut, "description": "循环设计的窗口尺寸超过交织矩阵尺寸"},
        404: {"model": ErrorOut, "description": "设计不存在"},
    },
    tags=["designs"],
)
def balance_hotspots(design_id: int, request: BalanceHotspotsIn) -> dict:
    """局部交织均衡审计：读取已入库设计并重新生成交织矩阵，按给定窗口逐窗
    统计 1（经纱在上）的占比。非循环设计只扫描完整落入矩阵的窗口（窗口大于
    矩阵时无窗可扫）；循环设计以每个原矩阵单元为左上角、两轴取模回绕，窗口
    不得超过矩阵尺寸（否则 400）。返回按左上角坐标稳定排序的超界窗口、占比
    及偏离最近阈值的幅度，并汇总最严重窗口（无超界时为 null）。接口只读
    计算，不改库。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    try:
        result = weave.scan_balance_hotspots(
            stored["threading"],
            stored["tie_up"],
            stored["treadling"],
            stored["cyclic"],
            request.window_rows,
            request.window_cols,
            request.warp_ratio_min,
            request.warp_ratio_max,
        )
    except weave.DesignError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "design_id": design_id,
        "cyclic": stored["cyclic"],
        **result.to_dict(),
    }


@app.post(
    "/designs/{design_id}/loom-fit",
    response_model=LoomFitOut,
    responses={404: {"model": ErrorOut, "description": "设计不存在"}},
    tags=["designs"],
)
def loom_fit(design_id: int, request: LoomFitIn) -> dict:
    """织机适配：读取已入库设计并重新生成交织矩阵，按**完整列向量相同的经纱
    归入同一约简综框、完整行向量相同的纬纱归入同一约简踏板**（等价类按首次
    出现顺序从 0 编号）重建 threading / tie_up / treadling，并逐点校验重建
    矩阵与原矩阵一致。返回不同列/行向量数——即织出同一矩阵所需的**最少
    综框 / 踏板数**、约简草案及相对原设计的减少量；仅当两项最少数量均不超过
    设备上限（max_shafts、max_treadles，1～64）时 `fits=true`。接口只读
    计算，不改库；设计不存在返回 404。"""
    stored = db.get_design(design_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"design {design_id} not found")
    reduction = weave.reduce_to_loom(
        stored["threading"],
        stored["tie_up"],
        stored["treadling"],
        stored["max_float"],
    )
    return {
        "design_id": design_id,
        "warp_ends": stored["warp_ends"],
        "weft_picks": stored["weft_picks"],
        "max_shafts": request.max_shafts,
        "max_treadles": request.max_treadles,
        "fits": (
            reduction.min_shafts <= request.max_shafts
            and reduction.min_treadles <= request.max_treadles
        ),
        "min_shafts": reduction.min_shafts,
        "min_treadles": reduction.min_treadles,
        "original_shafts": reduction.original_shafts,
        "original_treadles": reduction.original_treadles,
        "shafts_reduction": reduction.original_shafts - reduction.min_shafts,
        "treadles_reduction": reduction.original_treadles - reduction.min_treadles,
        "reduced_draft": {
            "threading": reduction.threading,
            "tie_up": reduction.tie_up,
            "treadling": reduction.treadling,
            "interlacing": reduction.interlacing,
        },
    }

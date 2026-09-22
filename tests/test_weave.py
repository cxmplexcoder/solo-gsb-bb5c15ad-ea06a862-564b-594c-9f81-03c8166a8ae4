"""核心织纹逻辑的单元测试（不依赖 FastAPI）。"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.weave import (  # noqa: E402
    DesignError,
    FloatViolation,
    aggregate_float_risk_zones,
    analyze_repeat_period,
    audit_floats,
    best_alignment,
    build_interlacing,
    reduce_to_loom,
    repeat_period_of_matrix,
    scan_balance_hotspots,
    search_repair_plans,
    suggest_tie_up_flips,
    threshold_profile,
    validate_design,
    _proper_divisors,
    _runs,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def assert_tiles(matrix, period, msg):
    """校验 period.unit 在经纬两方向取模平铺后逐点等于整幅矩阵。"""
    wp, wfp, unit = period.warp_period, period.weft_period, period.unit
    picks, ends = len(matrix), len(matrix[0])
    check(wp >= 1 and wfp >= 1, f"{msg}: periods positive")
    check(ends % wp == 0, f"{msg}: warp_period divides ends")
    check(picks % wfp == 0, f"{msg}: weft_period divides picks")
    check(len(unit) == wfp and all(len(r) == wp for r in unit),
          f"{msg}: unit is weft_period x warp_period")
    tiled = all(
        matrix[h][w] == unit[h % wfp][w % wp]
        for h in range(picks)
        for w in range(ends)
    )
    check(tiled, f"{msg}: unit tiles the full matrix seamlessly (mod wrap)")



def test_interlacing():
    # 平纹：2 综 2 踏
    threading = [0, 1, 0, 1]
    tie_up = [[1, 0], [0, 1]]
    treadling = [0, 1, 0, 1]
    m = build_interlacing(threading, tie_up, treadling)
    check(m == [[1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]], "plain weave interlacing")


def test_warp_and_weft_floats():
    # 第 0 根经纱连续 3 纬浮起；第 0 纬纬纱连续浮过 3 根经纱
    threading = [0, 0, 0]
    tie_up = [[1], [0]]
    treadling = [0, 0, 0, 1]
    # 矩阵:
    # 1 1 1
    # 1 1 1
    # 1 1 1
    # 0 0 0
    _, v = audit_floats(threading, tie_up, treadling, max_float=2, cyclic=False)
    warp = [x for x in v if x.direction == "warp"]
    weft = [x for x in v if x.direction == "weft"]
    check(len(warp) == 3 and all(x.length == 3 and x.start == 0 for x in warp),
          "warp floats of length 3 on all 3 ends")
    check(len(weft) == 1 and weft[0].yarn_index == 3 and weft[0].length == 3,
          "weft float of length 3 on pick 3")
    check(all(x.yarn_index in (0, 1, 2) for x in warp), "warp yarn indices")


def test_cyclic_merge():
    # 列: 1 1 0 0 1 1 -> 非循环两段(0,2),(4,2)；循环合并为 (4,4)
    check(_runs([1, 1, 0, 0, 1, 1], 1, False) == [(0, 2), (4, 2)],
          "non-cyclic runs")
    check(_runs([1, 1, 0, 0, 1, 1], 1, True) == [(4, 4)],
          "cyclic merge across boundary")
    # 全 1 列：只有一段，不自我合并
    check(_runs([1, 1, 1], 1, True) == [(0, 3)],
          "all-float line stays a single run")
    # 首尾不相接时不合并
    check(_runs([1, 0, 0, 1, 0], 1, True) == [(0, 1), (3, 1)],
          "no merge when boundary not both target")


def test_cyclic_audit_end_to_end():
    # 2/2 斜纹 4x4 循环，max_float=2：循环合并后每根经纱浮长恰为 2，不超限
    threading = [0, 1, 2, 3]
    tie_up = [[1, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 0, 1, 1] if False else [0, 0, 1, 1],
              [1, 0, 0, 1]]
    treadling = [0, 1, 2, 3]
    _, v = audit_floats(threading, tie_up, treadling, max_float=2, cyclic=True)
    check(v == [], "2/2 twill cyclic: no float exceeds 2")
    # 非循环时边界处浮长仍 <= 2，也不超限
    _, v2 = audit_floats(threading, tie_up, treadling, max_float=2, cyclic=False)
    check(v2 == [], "2/2 twill non-cyclic: no float exceeds 2")
    # max_float=1 时循环与非循环都应报出浮长 2
    _, v3 = audit_floats(threading, tie_up, treadling, max_float=1, cyclic=True)
    check(len(v3) > 0 and all(x.length == 2 for x in v3),
          "2/2 twill with max_float=1 flags length-2 floats")


def test_cyclic_wrap_violation():
    # 构造跨边界合并后才超限的情形：列 1 1 0 1 1，max_float=2
    # 非循环：两段长度 2、2，不超限；循环：合并为长度 4，超限
    threading = [0]
    tie_up = [[1], [0]]
    treadling = [0, 0, 1, 0, 0]
    _, v_off = audit_floats(threading, tie_up, treadling, max_float=2, cyclic=False)
    check(v_off == [], "wrap-around float not flagged when cyclic=False")
    _, v_on = audit_floats(threading, tie_up, treadling, max_float=2, cyclic=True)
    check(len(v_on) == 1 and v_on[0].direction == "warp"
          and v_on[0].yarn_index == 0 and v_on[0].start == 3 and v_on[0].length == 4,
          "wrap-around float merged to length 4 starting at 3")


def test_repeat_period_plain_weave():
    # 平纹 4x4：交织矩阵 [[1,0,1,0],[0,1,0,1],...]，经纬周期都是 2，
    # 单元 2x2，是严格二维重复
    threading = [0, 1, 0, 1]
    tie_up = [[1, 0], [0, 1]]
    treadling = [0, 1, 0, 1]
    period = analyze_repeat_period(threading, tie_up, treadling, 2)
    matrix = build_interlacing(threading, tie_up, treadling)
    check(period.warp_period == 2 and period.weft_period == 2,
          "plain weave: both periods are 2")
    check(period.unit == [[1, 0], [0, 1]], "plain weave: 2x2 unit")
    check(period.strict_2d_repeat is True, "plain weave: strict 2D repeat")
    assert_tiles(matrix, period, "plain weave")


def test_repeat_period_full_size_when_prime():
    # 3x3 满幅随机（尺寸为素数且无重复）：不存在小于整幅的周期，
    # 返回完整尺寸，unit 即整幅矩阵，strict_2d_repeat=False
    matrix = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]
    period = repeat_period_of_matrix(matrix)
    check(period.warp_period == 3 and period.weft_period == 3,
          "prime 3x3 matrix: no period smaller than full size")
    check(period.unit == matrix, "prime 3x3 matrix: unit is the full matrix")
    check(period.strict_2d_repeat is False,
          "prime 3x3 matrix: not strict 2D repeat")


def test_repeat_period_one_axis_only():
    # 仅经向重复、纬向不重复：列 [A B A B]，但两行不同
    # 1 0 1 0
    # 1 1 1 1
    # 经向周期 2（<4），纬向周期 2（=行数），不是严格二维重复
    matrix = [
        [1, 0, 1, 0],
        [1, 1, 1, 1],
    ]
    period = repeat_period_of_matrix(matrix)
    check(period.warp_period == 2, "one-axis: warp period 2")
    check(period.weft_period == 2, "one-axis: weft period full size 2")
    check(period.unit == [[1, 0], [1, 1]], "one-axis: unit is 2x2 (full height)")
    check(period.strict_2d_repeat is False,
          "one-axis: not strict 2D repeat (weft has no smaller period)")
    assert_tiles(matrix, period, "one-axis")


def test_repeat_period_based_on_matrix_not_threading():
    # 关键：周期必须基于交织矩阵，而非分别检查 threading/treadling。
    # 3 踏 3 综，仅踩踏板 0、1；tie_up 的第 0、2 列在这两行上相同
    # （穿综 0 与 2 的经纱交织表现一致），尽管穿综索引本身不同。
    # threading = [0,1,2,1]（作为序列最小周期 4，无 2-周期，因为 0!=2），
    # treadling = [0,1,0,1]（周期 2）。交织矩阵中穿综 0 与 2 的列相同，
    # 因此矩阵经向周期为 2。
    threading = [0, 1, 2, 1]
    tie_up = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    treadling = [0, 1, 0, 1]
    period = analyze_repeat_period(threading, tie_up, treadling, 4)
    matrix = build_interlacing(threading, tie_up, treadling)
    # threading 序列本身没有 2-周期（threading[0]=0 != threading[2]=2）
    check(any(threading[w] != threading[w % 2] for w in range(4)),
          "threading sequence alone has no 2-period")
    check(period.warp_period == 2,
          "matrix warp period 2 despite threading sequence period 4")
    check(period.weft_period == 2, "matrix weft period 2")
    check(period.unit == [[1, 0], [0, 1]], "matrix-derived 2x2 unit")
    check(period.strict_2d_repeat is True, "matrix-derived strict 2D repeat")
    assert_tiles(matrix, period, "matrix-based")


def test_repeat_period_cyclic_boundary():
    # 循环设计的边界：单列、treadling [0,0,1] 重复两次（6 纬）。
    # 列为 1 1 0 1 1 0，周期 3 整除 6；取模回绕比较必须正确跨越首尾边界，
    # 不能因为末段(1,1)接首段(1,1)看似相连而误判为更小周期（2 不整除 6）。
    threading = [0]
    tie_up = [[1], [0]]
    treadling = [0, 0, 1, 0, 0, 1]
    # 该设计按 cyclic=true 审计
    matrix, _ = audit_floats(threading, tie_up, treadling, 2, cyclic=True)
    period = analyze_repeat_period(threading, tie_up, treadling, 2)
    check(period.warp_period == 1, "single-column: warp period 1")
    check(period.weft_period == 3, "cyclic boundary: weft period 3 (not 2)")
    check(period.unit == [[1], [1], [0]], "cyclic boundary: 3x1 unit")
    check(period.strict_2d_repeat is False,
          "cyclic boundary single column: not strict 2D repeat")
    assert_tiles(matrix, period, "cyclic boundary")

    # 直接对跨边界平铺的矩阵验证回绕：4 列模式在边界处也必须对齐
    wrap_matrix = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    wp = repeat_period_of_matrix(wrap_matrix)
    check(wp.warp_period == 2 and wp.weft_period == 2,
          "wrap matrix aligned across both boundaries")
    assert_tiles(wrap_matrix, wp, "wrap matrix")


def test_repeat_period_constant_and_single_cell():
    # 全 1 矩阵 6x4：经纬周期都为 1
    all_ones = [[1] * 4 for _ in range(6)]
    period = repeat_period_of_matrix(all_ones)
    check(period.warp_period == 1 and period.weft_period == 1
          and period.unit == [[1]], "constant matrix: 1x1 unit")
    check(period.strict_2d_repeat is True, "constant 6x4: strict 2D repeat")
    assert_tiles(all_ones, period, "constant")

    # 1x1 单元即整幅，不是严格二维重复
    single = [[0]]
    p1 = repeat_period_of_matrix(single)
    check(p1.warp_period == 1 and p1.weft_period == 1 and p1.unit == [[0]],
          "single cell: periods 1")
    check(p1.strict_2d_repeat is False, "single cell: not strict 2D repeat")


def test_repeat_period_properties_random():
    # 随机交织矩阵：周期整除尺寸、单元无缝平铺、且没有更小的整除周期
    # （用朴素因子枚举交叉验证最小性）。
    rng = random.Random(20260921)
    check(_proper_divisors(12) == [1, 2, 3, 4, 6], "proper divisors helper")
    for picks in range(1, 9):
        for ends in range(1, 9):
            matrix = [[rng.randint(0, 1) for _ in range(ends)] for _ in range(picks)]
            period = repeat_period_of_matrix(matrix)
            assert_tiles(matrix, period, f"random {picks}x{ends}")

            for d in _proper_divisors(ends):
                if d >= period.warp_period:
                    break
                has_smaller = all(
                    matrix[h][w] == matrix[h][w % d]
                    for h in range(picks) for w in range(ends)
                )
                check(not has_smaller,
                      f"{picks}x{ends}: no warp divisor {d} smaller than "
                      f"{period.warp_period}")
            for d in _proper_divisors(picks):
                if d >= period.weft_period:
                    break
                has_smaller = all(
                    matrix[h][w] == matrix[h % d][w]
                    for h in range(picks) for w in range(ends)
                )
                check(not has_smaller,
                      f"{picks}x{ends}: no weft divisor {d} smaller than "
                      f"{period.weft_period}")
            expected_strict = period.warp_period < ends and period.weft_period < picks
            check(period.strict_2d_repeat == expected_strict,
                  f"{picks}x{ends}: strict_2d_repeat flag correct")


def test_tie_up_suggestions_basic():
    # 列 1 1 0 1 1 循环合并为长度 4 的经向浮长（max_float=2），共 1 条违规
    threading = [0]
    tie_up = [[1], [0]]
    treadling = [0, 0, 1, 0, 0]
    suggestions = suggest_tie_up_flips(threading, tie_up, treadling, 2, True)
    # 仅 flip tie_up[0][0] -> 0 可消除违规；受影响点 = 踩踏板 0 的 4 纬 × 1 经
    check(len(suggestions) == 1, "exactly one improving flip")
    s = suggestions[0]
    check(s.treadle == 0 and s.shaft == 0 and s.new_value == 0,
          "flip tie_up[0][0] to 0")
    check(s.remaining_violations == 0, "no violations remain after flip")
    check(s.affected_points == 4, "4 interlacing points affected")
    # 原设计不被修改
    check(tie_up == [[1], [0]], "original tie_up unchanged")


def test_tie_up_suggestions_ordering():
    # 3 根经纱各浮 3 纬 + 末纬纬浮 3 经，共 4 条违规；多个单步翻转可改善
    threading = [0, 1, 1]
    tie_up = [[1, 1], [0, 0]]
    treadling = [0, 0, 0, 1]
    suggestions = suggest_tie_up_flips(threading, tie_up, treadling, 2, False)
    got = [(s.treadle, s.shaft, s.remaining_violations, s.affected_points)
           for s in suggestions]
    # 按 (剩余违规数, 受影响点数, 踏板索引, 综框索引) 升序
    check(got == [(0, 1, 2, 6), (1, 0, 3, 1), (1, 1, 3, 2), (0, 0, 3, 3)],
          "suggestions sorted by remaining, affected, treadle, shaft")
    check(all(s.remaining_violations < 4 for s in suggestions),
          "all candidates strictly reduce violations")
    check(tie_up == [[1, 1], [0, 0]], "original tie_up unchanged")


def test_tie_up_suggestions_no_improvement():
    # 平纹无违规：违规数为 0，不存在严格更少的翻转，返回空列表
    threading = [0, 1, 0, 1]
    tie_up = [[1, 0], [0, 1]]
    treadling = [0, 1, 0, 1]
    check(suggest_tie_up_flips(threading, tie_up, treadling, 2, False) == [],
          "empty list when no flip improves")


def test_repair_plans_two_flips_needed():
    # 单列 6 纬全 1，cyclic 下整列一段长度 6，max_float=2：1 条经向违规。
    # 每个踏板各踩 1 纬，单次翻转只产生 1 个 0，仍有 1 条违规（长度 5）；
    # 循环上间隔恰为 3 的两次翻转把 1-段切成三段各长 2，违规清零。
    threading = [0]
    tie_up = [[1], [1], [1], [1], [1], [1]]
    treadling = [0, 1, 2, 3, 4, 5]
    _, base_v = audit_floats(threading, tie_up, treadling, 2, True)
    check(len(base_v) == 1, "baseline has exactly one violation")

    plans = search_repair_plans(threading, tie_up, treadling, 2, True,
                                max_changes=1)
    check(plans == [], "no improving plan within 1 flip")

    plans = search_repair_plans(threading, tie_up, treadling, 2, True,
                                max_changes=2)
    # 循环 6 点上两个 0 间隔恰为 3 时三段各长 2：恰 3 个方案
    check(len(plans) == 3, f"3 two-flip plans, got {len(plans)}")
    seq0 = [(s.treadle, s.shaft) for s in plans[0].steps]
    check(seq0 == [(0, 0), (3, 0)], "lexicographically smallest sequence first")
    check([(s.treadle, s.shaft) for s in plans[1].steps] == [(1, 0), (4, 0)]
          and [(s.treadle, s.shaft) for s in plans[2].steps] == [(2, 0), (5, 0)],
          "remaining plans in lexicographic order")
    for p in plans:
        check(p.changes == 2 and p.final_violations == 0,
              "every plan uses 2 flips and clears violations")
        check(p.changed_points == 2, "2 interlacing points changed")
        check(len(p.steps) == 2 and p.steps[0].step == 1 and p.steps[1].step == 2,
              "steps are numbered 1..2")
        check(p.steps[0].old_value == 1 and p.steps[0].new_value == 0
              and p.steps[1].old_value == 1 and p.steps[1].new_value == 0,
              "flips 1 -> 0 with correct old/new values")
        check(p.steps[0].violations_after == 1,
              "intermediate state may keep violations unchanged")
        check(p.violations == [], "final violation list empty")
        check(sum(x for row in p.interlacing for x in row) == 4,
              "final matrix has exactly two zeros")
    check(tie_up == [[1], [1], [1], [1], [1], [1]],
          "original tie_up unchanged")


def test_repair_plans_state_dedup():
    # 同一绑综终态可由不同翻转序列到达：按状态去重，只保留字典序最小序列。
    # 2 踏 2 综，单列单纬时仅被踩踏板所提升的综框影响矩阵；构造多步搜索，
    # 验证终态去重（无重复终态）且原设计不被修改。
    threading = [0, 1]
    tie_up = [[1, 1], [0, 0]]
    treadling = [0, 0, 0, 1]
    plans = search_repair_plans(threading, tie_up, treadling, 2, False,
                                max_changes=3, limit=50)
    finals = []
    for p in plans:
        check(p.final_violations < 4, "plan strictly improves over baseline")
        # 同一单元在一条序列内不重复出现
        coords = [(s.treadle, s.shaft) for s in p.steps]
        check(len(coords) == len(set(coords)),
              "no cell flipped twice within one sequence")
        finals.append(tuple(tuple(row) for row in p.interlacing))
    check(len(finals) == len(set(finals)),
          "equivalent final states deduplicated")
    # 排序键：最终违规数、改动数、变化点数、序列字典序
    keys = [
        (p.final_violations, p.changes, p.changed_points,
         tuple((s.treadle, s.shaft) for s in p.steps))
        for p in plans
    ]
    check(keys == sorted(keys), "plans sorted by the declared key")
    check(tie_up == [[1, 1], [0, 0]], "original tie_up unchanged")


def test_repair_plans_limit_and_validation():
    threading = [0]
    tie_up = [[1], [1], [1], [1], [1], [1]]
    treadling = [0, 1, 2, 3, 4, 5]
    plans = search_repair_plans(threading, tie_up, treadling, 2, True,
                                max_changes=2, limit=2)
    check(len(plans) == 2, "limit caps the number of plans")
    for bad in (0, 4):
        try:
            search_repair_plans(threading, tie_up, treadling, 2, True,
                                max_changes=bad)
            raise AssertionError(f"max_changes={bad} should be rejected")
        except DesignError:
            pass
    try:
        search_repair_plans(threading, tie_up, treadling, 2, True, limit=0)
        raise AssertionError("limit=0 should be rejected")
    except DesignError:
        pass
    print("  ok: repair plan parameter validation")


def test_best_alignment_shift_recovery():
    # b 为 a 沿纬向循环移 2 行、经向循环移 1 列：b[h][w] = a[(h+2)%3][(w+1)%3]
    # 对齐约定 aligned[h][w] = b[(h+ro)%H][(w+co)%W]，故最佳偏移为 (1, 2)
    a = [[1, 0, 1], [0, 1, 1], [1, 0, 0]]
    b = [[a[(h + 2) % 3][(w + 1) % 3] for w in range(3)] for h in range(3)]
    result = best_alignment(a, b)
    check(result.weft_offset == 1 and result.warp_offset == 2,
          "shifted copy: best offset recovers the cyclic shift")
    check(result.mismatch_count == 0 and result.mismatch_ratio == 0.0,
          "shifted copy: zero mismatches")
    check(result.aligned == a, "shifted copy: aligned matrix equals first matrix")
    check(result.mismatches == [], "shifted copy: no mismatch points")
    check(result.row_mismatches == [0, 0, 0]
          and result.col_mismatches == [0, 0, 0],
          "shifted copy: zero row/col summaries")
    check(a == [[1, 0, 1], [0, 1, 1], [1, 0, 0]]
          and b == [[0, 0, 1], [0, 1, 1], [1, 1, 0]],
          "input matrices unchanged")


def test_best_alignment_mismatch_details():
    # 仅 (2, 2) 单元不同：最佳对齐为零偏移，恰 1 处错位
    a = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    c = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
    result = best_alignment(a, c)
    check(result.weft_offset == 0 and result.warp_offset == 0,
          "single-cell diff: zero offset")
    check(result.mismatch_count == 1, "single-cell diff: one mismatch")
    check(abs(result.mismatch_ratio - 1 / 9) < 1e-12,
          "single-cell diff: ratio 1/9")
    check(len(result.mismatches) == 1, "single-cell diff: one mismatch point")
    m = result.mismatches[0]
    check((m.row, m.col, m.design_value, m.other_value) == (2, 2, 0, 1),
          "single-cell diff: mismatch coordinate and values")
    check(result.row_mismatches == [0, 0, 1]
          and result.col_mismatches == [0, 0, 1],
          "single-cell diff: row/col summaries")
    check(result.aligned == c, "single-cell diff: aligned is unshifted second matrix")


def test_best_alignment_tie_break():
    # 行全同：偏移 (0,1) 与 (1,1) 同为 0 错位，按行偏移升序取 (0,1)
    a = [[1, 0], [1, 0]]
    b = [[0, 1], [0, 1]]
    result = best_alignment(a, b)
    check((result.weft_offset, result.warp_offset) == (0, 1),
          "row tie broken by smaller weft offset")
    check(result.mismatch_count == 0, "row tie: zero mismatches")
    # 单行：偏移 1 与 3 同为 0 错位，按列偏移升序取 1
    a1 = [[1, 0, 1, 0]]
    b1 = [[0, 1, 0, 1]]
    r1 = best_alignment(a1, b1)
    check((r1.weft_offset, r1.warp_offset) == (0, 1),
          "column tie broken by smaller warp offset")
    # 全同常量矩阵：所有偏移平局，取 (0, 0)
    r2 = best_alignment([[1, 1], [1, 1]], [[1, 1], [1, 1]])
    check((r2.weft_offset, r2.warp_offset) == (0, 0)
          and r2.mismatch_count == 0,
          "constant matrices: offset (0, 0)")


def test_best_alignment_dimension_mismatch():
    for a, b in (
        ([[1, 0], [0, 1]], [[1, 0, 1], [0, 1, 0]]),   # 列数不同
        ([[1, 0], [0, 1]], [[1, 0]]),                 # 行数不同
        ([[1]], []),                                  # 空矩阵
    ):
        try:
            best_alignment(a, b)
            raise AssertionError(f"{a} vs {b} should be rejected")
        except DesignError:
            pass
    print("  ok: dimension mismatch raises DesignError")


def test_threshold_profile_cyclic_wrap():
    # 单列 1 1 0 1 1，cyclic 合并为长度 4 的经向浮长：
    # 阈值 1..3 各 1 条经向违规，阈值 4 起零违规
    threading = [0]
    tie_up = [[1], [0]]
    treadling = [0, 0, 1, 0, 0]
    profile = threshold_profile(threading, tie_up, treadling, True, 4)
    check(profile.max_float_limit == 4, "profile covers thresholds 1..4")
    check([p.max_float for p in profile.points] == [1, 2, 3, 4],
          "profile thresholds ascend 1..limit")
    check([(p.warp_violations, p.weft_violations, p.total_violations)
           for p in profile.points] == [(1, 0, 1), (1, 0, 1), (1, 0, 1), (0, 0, 0)],
          "cyclic wrap float cleared at threshold 4")
    check(profile.min_zero_violation_threshold == 4,
          "min zero-violation threshold is 4")
    # 上限内达不到零违规时为 None
    short = threshold_profile(threading, tie_up, treadling, True, 2)
    check(short.min_zero_violation_threshold is None
          and all(p.total_violations == 1 for p in short.points),
          "no zero-violation threshold within limit -> None")
    # 非循环语义下同一设计为两段长度 2 的浮长：阈值 1 有 2 条违规，阈值 2 清零
    plain = threshold_profile(threading, tie_up, treadling, False, 3)
    check(plain.min_zero_violation_threshold == 2
          and [p.total_violations for p in plain.points] == [2, 0, 0],
          "non-cyclic semantics preserved per threshold")


def test_threshold_profile_weft_direction():
    # 第 0 纬纬纱连续浮过 3 根经纱：阈值 1、2 各 1 条纬向违规，阈值 3 清零
    threading = [0, 0, 0]
    tie_up = [[0], [1]]
    treadling = [0, 1]
    profile = threshold_profile(threading, tie_up, treadling, False, 3)
    check([(p.warp_violations, p.weft_violations) for p in profile.points]
          == [(0, 1), (0, 1), (0, 0)],
          "weft-only float reported per threshold")
    check(profile.min_zero_violation_threshold == 3,
          "weft float cleared at threshold 3")


def test_threshold_profile_matches_audit_and_validation():
    # 随机设计：每档总数必须与同参数 audit_floats 的违规数一致，且单调不增
    rng = random.Random(20260922)
    for _ in range(30):
        shafts = rng.randint(1, 4)
        treadles = rng.randint(1, 4)
        threading = [rng.randrange(shafts) for _ in range(rng.randint(1, 8))]
        treadling = [rng.randrange(treadles) for _ in range(rng.randint(1, 8))]
        tie_up = [[rng.randint(0, 1) for _ in range(shafts)]
                  for _ in range(treadles)]
        cyclic = rng.random() < 0.5
        profile = threshold_profile(threading, tie_up, treadling, cyclic, 6)
        totals = []
        for point in profile.points:
            _, violations = audit_floats(
                threading, tie_up, treadling, point.max_float, cyclic
            )
            warp = sum(1 for v in violations if v.direction == "warp")
            check(point.total_violations == len(violations)
                  and point.warp_violations == warp
                  and point.weft_violations == len(violations) - warp,
                  "profile point matches full audit at the same threshold")
            totals.append(point.total_violations)
        check(totals == sorted(totals, reverse=True),
              "violation counts are non-increasing in threshold")
        if profile.min_zero_violation_threshold is not None:
            t = profile.min_zero_violation_threshold
            check(totals[t - 1] == 0 and (t == 1 or totals[t - 2] > 0),
                  "min zero-violation threshold is the first zero point")

    # 参数校验：非整数 / 越界的上限一律拒绝
    for bad in (0, 65, -1, True, 2.5, "3"):
        try:
            threshold_profile([0], [[1]], [0], False, bad)
            raise AssertionError(f"max_float_limit={bad!r} should be rejected")
        except DesignError:
            pass
    print("  ok: threshold profile parameter validation")


def test_balance_hotspots_non_cyclic():
    # 4x4 分块矩阵：
    # 1 1 0 0 / 1 1 0 0 / 0 0 1 1 / 0 0 1 1
    threading = [0, 0, 1, 1]
    tie_up = [[1, 0], [0, 1]]
    treadling = [0, 0, 1, 1]
    scan = scan_balance_hotspots(
        threading, tie_up, treadling, False, 2, 2, 0.25, 0.75
    )
    check(scan.windows_scanned == 9, "non-cyclic scans only contained windows")
    coords = [(h.row, h.col) for h in scan.hotspots]
    check(coords == [(0, 0), (0, 2), (2, 0), (2, 2)],
          "non-cyclic hotspots sorted by top-left coordinate")
    check([h.warp_ratio for h in scan.hotspots] == [1.0, 0.0, 0.0, 1.0],
          "non-cyclic hotspot ratios")
    check(all(h.deviation == 0.25 for h in scan.hotspots),
          "non-cyclic deviations from the nearest threshold")
    check(scan.worst_hotspot is scan.hotspots[0],
          "worst hotspot: tie broken by earliest scan order")

    # 阈值含边界：占比 0 / 1 在 [0, 1] 内不超界
    scan = scan_balance_hotspots(
        threading, tie_up, treadling, False, 2, 2, 0.0, 1.0
    )
    check(scan.hotspots == [] and scan.worst_hotspot is None,
          "inclusive thresholds: no hotspots, worst is None")

    # 窗口大于矩阵：非循环无窗可扫
    scan = scan_balance_hotspots(
        threading, tie_up, treadling, False, 5, 5, 0.25, 0.75
    )
    check(scan.windows_scanned == 0 and scan.hotspots == []
          and scan.worst_hotspot is None,
          "non-cyclic window larger than matrix scans nothing")


def test_balance_hotspots_cyclic_wrap():
    # 3x3 上三角矩阵：1 1 1 / 0 1 0 / 0 0 1
    threading = [0, 1, 2]
    tie_up = [[1, 1, 1], [0, 1, 0], [0, 0, 1]]
    treadling = [0, 1, 2]
    scan = scan_balance_hotspots(
        threading, tie_up, treadling, True, 2, 2, 0.3, 0.7
    )
    check(scan.windows_scanned == 9, "cyclic scans one window per cell")
    coords = [(h.row, h.col) for h in scan.hotspots]
    check(coords == [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)],
          "cyclic hotspots include wrap-around windows")
    check([h.warp_ratio for h in scan.hotspots]
          == [0.75, 0.75, 0.25, 0.25, 0.75, 0.75],
          "cyclic wrap-around window ratios")

    # 循环设计窗口超过矩阵尺寸 -> DesignError
    for rows, cols in ((4, 2), (2, 4), (4, 4)):
        try:
            scan_balance_hotspots(
                threading, tie_up, treadling, True, rows, cols, 0.0, 1.0
            )
            raise AssertionError(
                f"cyclic window {rows}x{cols} should be rejected"
            )
        except DesignError:
            pass
    print("  ok: cyclic window larger than matrix is rejected")


def test_balance_hotspots_random_against_brute_force():
    # 随机设计：与逐窗暴力求和对比，占比与扫描数必须一致
    rng = random.Random(20260922)
    for _ in range(30):
        shafts = rng.randint(1, 4)
        treadles = rng.randint(1, 4)
        threading = [rng.randrange(shafts) for _ in range(rng.randint(1, 8))]
        treadling = [rng.randrange(treadles) for _ in range(rng.randint(1, 8))]
        tie_up = [[rng.randint(0, 1) for _ in range(shafts)]
                  for _ in range(treadles)]
        cyclic = rng.random() < 0.5
        matrix = build_interlacing(threading, tie_up, treadling)
        picks, ends = len(matrix), len(matrix[0])
        wr = rng.randint(1, picks if cyclic else picks + 2)
        wc = rng.randint(1, ends if cyclic else ends + 2)
        lo = rng.random()
        hi = lo + rng.random() * (1.0 - lo)
        scan = scan_balance_hotspots(
            threading, tie_up, treadling, cyclic, wr, wc, lo, hi
        )

        if cyclic:
            origins = [(r, c) for r in range(picks) for c in range(ends)]
        else:
            origins = [(r, c)
                       for r in range(max(picks - wr + 1, 0))
                       for c in range(max(ends - wc + 1, 0))]
        check(scan.windows_scanned == len(origins),
              "scanned window count matches brute force origins")

        expected = []
        for r, c in origins:
            ones = sum(
                matrix[(r + i) % picks][(c + j) % ends]
                for i in range(wr) for j in range(wc)
            )
            ratio = ones / (wr * wc)
            if ratio < lo or ratio > hi:
                expected.append((r, c, ratio))
        got = [(h.row, h.col, h.warp_ratio) for h in scan.hotspots]
        check(len(got) == len(expected)
              and all(g == e for g, e in zip(got, expected)),
              "hotspot coordinates and ratios match brute force")
        for h in scan.hotspots:
            nearest = lo if h.warp_ratio < lo else hi
            check(abs(h.deviation - abs(h.warp_ratio - nearest)) < 1e-12,
                  "deviation is the distance to the nearest threshold")
        if expected:
            worst_dev = max(
                min(abs(ratio - lo), abs(ratio - hi))
                for _, _, ratio in expected
            )
            check(scan.worst_hotspot is not None
                  and abs(scan.worst_hotspot.deviation - worst_dev) < 1e-12,
                  "worst hotspot has the maximal deviation")
        else:
            check(scan.worst_hotspot is None,
                  "no hotspots -> worst is None")
    print("  ok: balance hotspots match brute force on random designs")


def test_balance_hotspots_validation():
    base = ([0], [[1]], [0])
    bad_calls = [
        (0, 2, 0.0, 1.0), (65, 2, 0.0, 1.0), (2, 0, 0.0, 1.0),
        (2, 65, 0.0, 1.0), (True, 2, 0.0, 1.0), (2.5, 2, 0.0, 1.0),
        (2, 2, -0.1, 1.0), (2, 2, 0.0, 1.1), (2, 2, 0.75, 0.25),
        (2, 2, True, 1.0), (2, 2, 0.0, "1"),
    ]
    for wr, wc, lo, hi in bad_calls:
        try:
            scan_balance_hotspots(*base, False, wr, wc, lo, hi)
            raise AssertionError(
                f"window=({wr},{wc}) thresholds=({lo},{hi}) should be rejected"
            )
        except DesignError:
            pass
    print("  ok: balance hotspots parameter validation")


def test_loom_fit_no_reduction_when_columns_distinct():
    # 2/2 斜纹 4 综 4 踏：交织矩阵的 4 个列向量、4 个行向量都互异，无法约简
    threading = [0, 1, 2, 3, 0, 1, 2, 3]
    tie_up = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
    treadling = [0, 1, 2, 3, 0, 1, 2, 3]
    red = reduce_to_loom(threading, tie_up, treadling, 2)
    check(red.original_shafts == 4 and red.original_treadles == 4,
          "twill: reports original counts")
    check(red.min_shafts == 4 and red.min_treadles == 4,
          "twill: four distinct column/row vectors => 4 shafts, 4 treadles")
    check(red.threading == threading and red.treadling == treadling
          and red.tie_up == tie_up,
          "twill: reduced draft equals the original draft")
    check(red.interlacing == build_interlacing(threading, tie_up, treadling),
          "twill: rebuilt matrix matches the original")


def test_loom_fit_merges_redundant_shafts_and_treadles():
    # 4 综 4 踏的穿法，但绑综列 0/2 相同、1/3 相同，行 0/2 相同、1/3 相同：
    # 交织矩阵只有 2 种列向量和 2 种行向量，本质是平纹
    threading = [0, 1, 2, 3, 0, 1, 2, 3]
    tie_up = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    treadling = [0, 1, 2, 3, 0, 1, 2, 3]
    red = reduce_to_loom(threading, tie_up, treadling, 2)
    check(red.min_shafts == 2 and red.min_treadles == 2,
          "redundant draft: 2 distinct column/row vectors => 2x2 loom")
    check(red.threading == [0, 1, 0, 1, 0, 1, 0, 1],
          "redundant draft: columns merged by full column vector, first-seen order")
    check(red.treadling == [0, 1, 0, 1, 0, 1, 0, 1],
          "redundant draft: rows merged by full row vector, first-seen order")
    check(red.tie_up == [[1, 0], [0, 1]],
          "redundant draft: tie-up rebuilt as plain weave")
    check(red.interlacing == build_interlacing(threading, tie_up, treadling),
          "redundant draft: rebuilt matrix pointwise equals the original")


def test_loom_fit_unused_shaft_but_distinct_columns():
    # 3 综只用 2 个：原绑综列 0、1 互异，约简后仍需 2 综（仅去掉未使用综框）
    threading = [0, 1, 0, 1, 0, 1]
    tie_up = [[1, 0, 0], [0, 1, 0]]
    treadling = [0, 1, 0, 1]
    red = reduce_to_loom(threading, tie_up, treadling, 2)
    check((red.original_shafts, red.original_treadles) == (3, 2),
          "unused shaft: original is 3 shafts x 2 treadles")
    check((red.min_shafts, red.min_treadles) == (2, 2),
          "unused shaft: minimum keeps the 2 used distinct columns")
    check(red.threading == [0, 1, 0, 1, 0, 1]
          and red.treadling == [0, 1, 0, 1]
          and red.tie_up == [[1, 0], [0, 1]],
          "unused shaft: draft rebuilt without the empty shaft")
    check(red.interlacing == build_interlacing(threading, tie_up, treadling),
          "unused shaft: rebuilt matrix matches")


def test_loom_fit_constant_matrix_collapses_to_one_by_one():
    # 绑综恒为 1：所有列、行向量各自相同 -> 1 综 1 踏
    threading = [0, 1, 2]
    tie_up = [[1, 1, 1], [1, 1, 1]]
    treadling = [0, 1, 0, 1]
    red = reduce_to_loom(threading, tie_up, treadling, 4)
    check((red.min_shafts, red.min_treadles) == (1, 1),
          "constant matrix collapses to 1 shaft x 1 treadle")
    check(red.threading == [0, 0, 0] and red.treadling == [0, 0, 0, 0]
          and red.tie_up == [[1]],
          "constant matrix: reduced draft is all zeros with 1x1 tie-up")
    check(red.interlacing == [[1] * 3 for _ in range(4)],
          "constant matrix: rebuilt matrix all ones")


def test_loom_fit_first_seen_numbering_and_single_cell():
    # 列向量按经纱首次出现顺序、行向量按纬纱首次出现顺序编号
    threading = [0, 0, 1, 1]
    tie_up = [[1, 0], [0, 1], [1, 0]]
    treadling = [2, 2, 0, 1, 1]
    red = reduce_to_loom(threading, tie_up, treadling, 5)
    matrix = build_interlacing(threading, tie_up, treadling)
    # 列 0=[1,1,1,0,0]、列 2=[0,0,0,1,1]：首次出现顺序 0,1
    # 行按出现顺序：行0=行1=行2=[1,1,0,0]（首个），行3=[0,0,1,1]
    check(red.threading == [0, 0, 1, 1],
          "first-seen: shaft labels follow column first appearance")
    check(red.treadling == [0, 0, 0, 1, 1],
          "first-seen: treadle labels follow row first appearance")
    check(red.tie_up == [[1, 0], [0, 1]],
          "first-seen: tie-up uses representative row/column values")
    check(red.interlacing == matrix, "first-seen: rebuilt matrix matches")

    # 1x1 矩阵：1 综 1 踏
    tiny = reduce_to_loom([0], [[0]], [0], 1)
    check((tiny.min_shafts, tiny.min_treadles) == (1, 1)
          and tiny.tie_up == [[0]] and tiny.interlacing == [[0]],
          "single-cell design: 1x1 reduced draft")


def test_loom_fit_random_properties():
    rng = random.Random(20260922)
    for _ in range(200):
        shafts = rng.randint(1, 5)
        treadles = rng.randint(1, 5)
        ends = rng.randint(1, 12)
        picks = rng.randint(1, 12)
        threading = [rng.randrange(shafts) for _ in range(ends)]
        treadling = [rng.randrange(treadles) for _ in range(picks)]
        tie_up = [[rng.randrange(2) for _ in range(shafts)]
                  for _ in range(treadles)]
        matrix = build_interlacing(threading, tie_up, treadling)
        red = reduce_to_loom(threading, tie_up, treadling, 1)

        # 最少数量 = 不同完整列/行向量数，且不超过原数量
        distinct_cols = len({tuple(matrix[h][w] for h in range(picks))
                             for w in range(ends)})
        distinct_rows = len({tuple(row) for row in matrix})
        check(red.min_shafts == distinct_cols,
              "random: min_shafts equals number of distinct column vectors")
        check(red.min_treadles == distinct_rows,
              "random: min_treadles equals number of distinct row vectors")
        check(red.min_shafts <= red.original_shafts
              and red.min_treadles <= red.original_treadles,
              "random: reduced counts never exceed original counts")

        # 约简草案尺寸自洽
        check(len(red.tie_up) == red.min_treadles
              and all(len(r) == red.min_shafts for r in red.tie_up),
              "random: reduced tie-up is min_treadles x min_shafts")
        check(all(0 <= s < red.min_shafts for s in red.threading)
              and all(0 <= t < red.min_treadles for t in red.treadling),
              "random: reduced indexes in range")
        # 每个约简综框/踏板都被实际使用
        check(sorted(set(red.threading)) == list(range(red.min_shafts))
              and sorted(set(red.treadling)) == list(range(red.min_treadles)),
              "random: every reduced shaft/treadle is used")
        # 逐点重建与原矩阵一致
        check(red.interlacing == matrix,
              "random: rebuilt matrix pointwise equals the original")
    print("  ok: loom fit random property checks (200 cases)")


def test_float_risk_zones_orthogonal_merge():
    # 经向竖段与纬向横段在一个单元处正交相邻 -> 合并为一个 L 形区域
    matrix = [[0] * 4 for _ in range(4)]
    violations = [
        FloatViolation("warp", yarn_index=1, start=0, length=3),
        FloatViolation("weft", yarn_index=2, start=1, length=3),
    ]
    zones = aggregate_float_risk_zones(matrix, violations, cyclic=False)
    check(len(zones) == 1, "orthogonal warp/weft runs merge into one zone")
    zone = zones[0]
    expected_cells = [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3)]
    check([(c.row, c.col) for c in zone.cells] == expected_cells,
          "merged zone cells in row-major order")
    check(zone.area == 5, "zone area counts unique cells")
    check(zone.max_float_length == 3, "max float length across violations")
    check(zone.directions == ["warp", "weft"], "directions ordered warp then weft")
    check(
        zone.min_row == 0 and zone.max_row == 2
        and zone.min_col == 1 and zone.max_col == 3
        and zone.height == 3 and zone.width == 3
        and zone.crosses_weft_seam is False and zone.crosses_warp_seam is False,
        "non-cyclic zone bbox is a plain bounding box",
    )


def test_float_risk_zones_diagonal_not_connected():
    # 两条 3 长浮长仅在对角方向接触 -> 两个独立区域
    matrix = [[0] * 4 for _ in range(4)]
    violations = [
        FloatViolation("warp", yarn_index=0, start=0, length=3),
        FloatViolation("weft", yarn_index=3, start=1, length=3),
    ]
    zones = aggregate_float_risk_zones(matrix, violations, cyclic=False)
    check(len(zones) == 2, "diagonal contact does not connect zones")
    lengths = [z.max_float_length for z in zones]
    check(lengths == [3, 3], "both zones keep float length 3")
    cells = [[(c.row, c.col) for c in z.cells] for z in zones]
    # 面积相同（3=3）时平局按最小单元坐标行主序：(0,0) 在 (3,1) 前
    check(cells[0] == [(0, 0), (1, 0), (2, 0)], "first zone is the earlier-seeded one")
    check(cells[1] == [(3, 1), (3, 2), (3, 3)], "second zone stays separate")

    # 不同严重度：4 长区域即使种子更晚也排在 3 长区域之前（5x5 上二者分离）
    big = [[0] * 5 for _ in range(5)]
    zones = aggregate_float_risk_zones(
        big,
        [FloatViolation("warp", 0, 0, 3), FloatViolation("weft", 4, 1, 4)],
        cyclic=False,
    )
    check([z.max_float_length for z in zones] == [4, 3],
          "zones sorted by max float length descending")
    check(zones[0].area == 4 and zones[1].area == 3, "areas follow sorting")


def test_float_risk_zones_cyclic_seams():
    # 4 列环面：行 0 上 start=3 的 3 长纬浮覆盖 (0,3)(0,0)(0,1)
    matrix = [[0] * 4 for _ in range(4)]
    v = [FloatViolation("weft", 0, 3, 3)]

    # 非循环下同条记录 start+length 越过矩阵边界 -> 拒绝
    raised = False
    try:
        aggregate_float_risk_zones(matrix, v, cyclic=False)
    except DesignError:
        raised = True
    check(raised, "non-cyclic run past the matrix edge is rejected")

    zones = aggregate_float_risk_zones(matrix, v, cyclic=True)
    zone = zones[0]
    check([(c.row, c.col) for c in zone.cells] == [(0, 0), (0, 1), (0, 3)],
          "wrap run covers tail+head cells in row-major order")
    check(zone.crosses_warp_seam is True and zone.crosses_weft_seam is False,
          "zone crossing last/first column flags warp seam only")
    check(zone.min_col == 0 and zone.max_col == 3 and zone.width == 3,
          "wrapping bbox: raw extremes touch both edges, wrapped width 3")
    check(zone.min_row == 0 and zone.max_row == 0 and zone.height == 1,
          "non-wrapped axis has a degenerate bbox")

    # 跨越首末行接缝的经向浮长：start=3 长 3 -> (3,0)(0,0)(1,0)
    zones = aggregate_float_risk_zones(
        matrix, [FloatViolation("warp", 0, 3, 3)], cyclic=True
    )
    zone = zones[0]
    check(zone.crosses_weft_seam is True and zone.crosses_warp_seam is False,
          "wrap along rows flags weft seam only")
    check(zone.min_row == 0 and zone.max_row == 3 and zone.height == 3,
          "wrapping row bbox: raw extremes touch both edges, wrapped height 3")

    # 非循环下分列于首末列（列 0 与列 3）的两个边界段不相连；
    # 循环时两列经首末列接缝相邻，合并为一区
    boundary = [FloatViolation("warp", 0, 2, 2), FloatViolation("warp", 3, 2, 2)]
    separate = aggregate_float_risk_zones(matrix, boundary, cyclic=False)
    check(len(separate) == 2, "first/last column runs stay separate when non-cyclic")
    joined = aggregate_float_risk_zones(matrix, boundary, cyclic=True)
    check(len(joined) == 1 and joined[0].crosses_warp_seam is True
          and joined[0].area == 4,
          "first/last column runs join across the warp seam when cyclic")


def test_float_risk_zones_cyclic_double_seam():
    # 整幅 4x4 全部风险：环面上连通为一区，两方向都跨缝，覆盖整个矩阵
    matrix = [[1] * 4 for _ in range(4)]
    violations = [
        FloatViolation("warp", w, 0, 4) for w in range(4)
    ] + [
        FloatViolation("weft", h, 0, 4) for h in range(4)
    ]
    zones = aggregate_float_risk_zones(matrix, violations, cyclic=True)
    check(len(zones) == 1, "all-marked torus is a single connected zone")
    zone = zones[0]
    check(zone.area == 16, "zone covers the whole 4x4 matrix")
    check(zone.crosses_weft_seam and zone.crosses_warp_seam,
          "full-matrix zone crosses both seams")
    check(zone.height == 4 and zone.width == 4, "wrapped bbox covers full size")
    check(zone.max_float_length == 4 and zone.directions == ["warp", "weft"],
          "full-matrix zone aggregates length and both directions")

    # 尺寸 1 的环面：上下左右都是自环，不应误判为跨缝
    zones = aggregate_float_risk_zones(
        [[1]], [FloatViolation("warp", 0, 0, 1)], cyclic=True
    )
    check(len(zones) == 1 and zones[0].area == 1
          and not zones[0].crosses_weft_seam and not zones[0].crosses_warp_seam
          and zones[0].height == 1 and zones[0].width == 1,
          "1x1 cyclic self-loop is one cell with no seam crossing")


def test_float_risk_zones_random_against_brute_force():
    """随机标记单元（用 wrap/非 wrap 的 FloatViolation 展开）与独立环面/平面
    洪水填充对比：区域数、成员、跨缝标志与包围盒尺寸逐例一致。"""
    rng = random.Random(20260922)

    def brute_force(marked, picks, ends, cyclic):
        remaining = set(marked)
        components = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            seen = {seed}
            queue = [seed]
            crosses = [False, False]
            while queue:
                r, c = queue.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if cyclic:
                        if dr != 0:
                            if picks == 1:
                                continue
                            nr, nc = (r + dr) % picks, c
                            wrapped = {r, nr} == {0, picks - 1}
                        else:
                            if ends == 1:
                                continue
                            nr, nc = r, (c + dc) % ends
                            wrapped = {c, nc} == {0, ends - 1}
                    else:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < picks and 0 <= nc < ends):
                            continue
                    if (nr, nc) not in marked:
                        continue
                    # 接缝边只在两个端点同属本分量时计数（另一端已被本
                    # 分量发现或尚待本分量发现）；通向其他分量的边不算
                    if cyclic and wrapped and (
                        (nr, nc) in seen or (nr, nc) in remaining
                    ):
                        if dr:
                            crosses[0] = True
                        if dc:
                            crosses[1] = True
                    if (nr, nc) in remaining:
                        remaining.remove((nr, nc))
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            components.append((seen, tuple(crosses)))
        return components

    for _ in range(300):
        picks = rng.randint(1, 6)
        ends = rng.randint(1, 6)
        cyclic = rng.random() < 0.5
        matrix = [[0] * ends for _ in range(picks)]
        violations = []
        marked = {}
        n_runs = rng.randint(0, picks + ends)
        for _n in range(n_runs):
            if rng.random() < 0.5:
                w = rng.randrange(ends)
                if cyclic:
                    start = rng.randrange(picks)
                    length = rng.randint(1, picks)
                else:
                    length = rng.randint(1, picks)
                    start = rng.randrange(picks - length + 1)
                violations.append(FloatViolation("warp", w, start, length))
                for k in range(length):
                    r = (start + k) if not cyclic else (start + k) % picks
                    marked[(r, w)] = True
            else:
                h = rng.randrange(picks)
                if cyclic:
                    start = rng.randrange(ends)
                    length = rng.randint(1, ends)
                else:
                    length = rng.randint(1, ends)
                    start = rng.randrange(ends - length + 1)
                violations.append(FloatViolation("weft", h, start, length))
                for k in range(length):
                    c = (start + k) if not cyclic else (start + k) % ends
                    marked[(h, c)] = True

        zones = aggregate_float_risk_zones(matrix, violations, cyclic)
        comps = brute_force(marked, picks, ends, cyclic)

        check(len(zones) == len(comps),
              f"random: zone count matches brute force ({picks}x{ends}, "
              f"cyclic={cyclic})")

        zone_cells = {frozenset((c.row, c.col) for c in z.cells) for z in zones}
        comp_cells = {frozenset(seen) for seen, _ in comps}
        check(zone_cells == comp_cells,
              "random: zone membership partitions match brute force")

        zone_by_cells = {
            frozenset((c.row, c.col) for c in z.cells): z for z in zones
        }
        for seen, crosses in comps:
            z = zone_by_cells[frozenset(seen)]
            check(z.area == len(seen), "random: area matches component size")
            check((z.crosses_weft_seam, z.crosses_warp_seam) == crosses,
                  "random: seam flags match brute-force wrap edges")
            if not crosses[0]:
                rows = [r for r, _ in seen]
                check(z.height == max(rows) - min(rows) + 1
                      and z.min_row == min(rows) and z.max_row == max(rows),
                      "random: non-wrapping row bbox")
            else:
                check(1 <= z.height <= picks,
                      "random: wrapping row height within matrix size")
            if not crosses[1]:
                cols = [c for _, c in seen]
                check(z.width == max(cols) - min(cols) + 1
                      and z.min_col == min(cols) and z.max_col == max(cols),
                      "random: non-wrapping col bbox")
            else:
                check(1 <= z.width <= ends,
                      "random: wrapping col width within matrix size")

        # 严重度排序键单调不增
        keys = [(z.max_float_length, z.area) for z in zones]
        check(keys == sorted(keys, key=lambda k: (-k[0], -k[1])),
              "random: zones severity-sorted")

    check(True, "random risk-zone property checks (300 cases)")


def test_float_risk_zones_empty_and_validation():
    matrix = [[0, 1], [1, 0]]
    check(aggregate_float_risk_zones(matrix, [], cyclic=False) == [],
          "no violations -> empty zone list")
    check(aggregate_float_risk_zones(matrix, [], cyclic=True) == [],
          "no violations cyclic -> empty zone list")

    # dict 形式的违规记录同样可展开
    zones = aggregate_float_risk_zones(
        matrix,
        [{"direction": "weft", "yarn_index": 0, "start": 0, "length": 2}],
        cyclic=False,
    )
    check([(c.row, c.col) for c in zones[0].cells] == [(0, 0), (0, 1)],
          "dict-shaped violations are accepted")

    bad_cases = [
        ([], [], False),                                   # 空矩阵
        ([[0], [0, 1]], [], False),                        # 非矩形矩阵
        (matrix, [FloatViolation("sideways", 0, 0, 1)], False),  # 方向非法
        (matrix, [FloatViolation("warp", 9, 0, 1)], False),      # 索引越界
        (matrix, [FloatViolation("warp", 0, 0, 0)], False),      # 长度非正
        (matrix, [FloatViolation("weft", 0, 0, 3)], False),      # 非循环越界
    ]
    for m, vs, cyc in bad_cases:
        try:
            aggregate_float_risk_zones(m, vs, cyc)
            check(False, f"invalid case rejected: {vs}")
        except DesignError:
            pass
    check(True, "invalid matrix/violation inputs raise DesignError")

    # 与真实审计一致：非循环 3x3 全 1（底行 0）矩阵的风险单元展开
    threading = [0, 0, 0]
    tie_up = [[1], [0]]
    treadling = [0, 0, 0, 1]
    m, vs = audit_floats(threading, tie_up, treadling, max_float=2, cyclic=False)
    zones = aggregate_float_risk_zones(m, vs, cyclic=False)
    # 三条经向 3 长覆盖前 3 行 3x3=9 个单元；底纬一条 3 长 (3,0..2) 与列段在
    # 第 2/3 行交界处上下相邻，全部 12 个矩阵单元连通
    check(len(zones) == 1 and zones[0].area == 12,
          "audit violations expand into one connected zone covering all 12 cells")
    check(zones[0].max_float_length == 3
          and zones[0].directions == ["warp", "weft"]
          and zones[0].height == 4 and zones[0].width == 3,
          "zone from real audit merges warp block and weft run")


def test_validation():
    ok_cases = []
    # 维度不一致：绑综矩阵不是矩形
    try:
        validate_design([0], [[1, 0], [1]], [0], 2)
        ok_cases.append(False)
    except DesignError:
        ok_cases.append(True)
    # 穿综索引越界
    try:
        validate_design([0, 2], [[1, 0], [0, 1]], [0], 2)
        ok_cases.append(False)
    except DesignError:
        ok_cases.append(True)
    # 踏综索引越界
    try:
        validate_design([0], [[1], [0]], [2], 2)
        ok_cases.append(False)
    except DesignError:
        ok_cases.append(True)
    # 绑综取值非 0/1
    try:
        validate_design([0], [[2]], [0], 2)
        ok_cases.append(False)
    except DesignError:
        ok_cases.append(True)
    # 空输入
    try:
        validate_design([], [[1]], [0], 2)
        ok_cases.append(False)
    except DesignError:
        ok_cases.append(True)
    # 负索引
    try:
        validate_design([-1], [[1]], [0], 2)
        ok_cases.append(False)
    except DesignError:
        ok_cases.append(True)
    check(all(ok_cases), "invalid designs are rejected")


if __name__ == "__main__":
    test_interlacing()
    test_warp_and_weft_floats()
    test_cyclic_merge()
    test_cyclic_audit_end_to_end()
    test_cyclic_wrap_violation()
    test_repeat_period_plain_weave()
    test_repeat_period_full_size_when_prime()
    test_repeat_period_one_axis_only()
    test_repeat_period_based_on_matrix_not_threading()
    test_repeat_period_cyclic_boundary()
    test_repeat_period_constant_and_single_cell()
    test_repeat_period_properties_random()
    test_tie_up_suggestions_basic()
    test_tie_up_suggestions_ordering()
    test_tie_up_suggestions_no_improvement()
    test_repair_plans_two_flips_needed()
    test_repair_plans_state_dedup()
    test_repair_plans_limit_and_validation()
    test_best_alignment_shift_recovery()
    test_best_alignment_mismatch_details()
    test_best_alignment_tie_break()
    test_best_alignment_dimension_mismatch()
    test_threshold_profile_cyclic_wrap()
    test_threshold_profile_weft_direction()
    test_threshold_profile_matches_audit_and_validation()
    test_balance_hotspots_non_cyclic()
    test_balance_hotspots_cyclic_wrap()
    test_balance_hotspots_random_against_brute_force()
    test_balance_hotspots_validation()
    test_loom_fit_no_reduction_when_columns_distinct()
    test_loom_fit_merges_redundant_shafts_and_treadles()
    test_loom_fit_unused_shaft_but_distinct_columns()
    test_loom_fit_constant_matrix_collapses_to_one_by_one()
    test_loom_fit_first_seen_numbering_and_single_cell()
    test_loom_fit_random_properties()
    test_float_risk_zones_orthogonal_merge()
    test_float_risk_zones_diagonal_not_connected()
    test_float_risk_zones_cyclic_seams()
    test_float_risk_zones_cyclic_double_seam()
    test_float_risk_zones_random_against_brute_force()
    test_float_risk_zones_empty_and_validation()
    test_validation()
    print("all tests passed")

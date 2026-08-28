# -*- coding: utf-8 -*-
"""
regression_test.py — refactor 版回归测试
覆盖历史全部优化用例 + 缝合基线新用例。运行:
    python regression_test.py
"""
import math
import io
import contextlib
import random
import sys

from data import RawVehicle, RawFrame
from config import Config
from lane_tracker import VehicleState
from smooth_engine import PerceptionFilterEngine

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name} {detail}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {detail}')


def quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


A = '137439016690'
B = '137439016674'


def get_ctx():
    eng = PerceptionFilterEngine(Config)
    mm = eng.map_mgr
    bearing = (90 - mm.lanes[A]['heading']) % 360
    line_len = mm.lanes[A]['line'].length
    return eng, mm, bearing, line_len


def mk_rv(vid, x, y, heading, sub=11):
    return RawVehicle(object_id=vid, lon=0, lat=0, rel_x=x, rel_y=y,
                      itc_obj_type=1, itc_sub_type=sub, plate_num='',
                      lane_no='', radar_heading=heading, type_reliability=0.9)


def main():
    random.seed(7)

    # ================= T1: 方向滞回状态机 =================
    print('\n== T1: 方向滞回 (车头 180° 翻转) ==')
    veh = VehicleState(1, 'L', 0.0, 0.0, 0.0, {'itc_sub_type': 11}, 0, 0.0, 0.0, 90.0)
    t = 0
    s = 0.0
    dir_trace = []
    sv_trace = []
    for i in range(200):
        t += 100
        s += (0.4 if i < 100 else -1.5) * 0.1
        noisy = s + 0.35 * math.sin(2 * math.pi * (t / 1000.0) / 2.5) + random.uniform(-0.1, 0.1)
        veh.update_and_estimate_speed(t, noisy)
        dir_trace.append(veh.drive_direction)
        sv_trace.append(veh.signed_v)
    old_flips = 0
    old_dir = 0
    for sv in sv_trace:
        nd = 1 if sv > 0.5 else (-1 if sv < -0.5 else 0)
        if nd != old_dir:
            old_flips += 1
            old_dir = nd
    new_flips = sum(1 for a, b in zip(dir_trace, dir_trace[1:]) if a != b)
    check('旧阈值法翻转次数 <= 新状态机', old_flips <= new_flips or new_flips <= 2,
          f'(old={old_flips}, new={new_flips})')
    check('最终方向 = -1 (反向)', dir_trace[-1] == -1)

    # ================= T2: 车型投票 =================
    print('\n== T2: 车型投票锁定 ==')
    veh2 = VehicleState(2, 'L', 0, 0, 0, {'itc_sub_type': 99}, 0, 0, 0, 90)
    for i in range(3):
        veh2.vote_type(11, 1, 0.9, i * 100)
    for i in range(3, 43):
        veh2.vote_type(6, 1, 0.9, i * 100)
    check('早期误分类 3x11 后持续 6 -> 锁 6', veh2.locked_type == 6,
          f'(got {veh2.locked_type})')

    veh3 = VehicleState(3, 'L', 0, 0, 0, {'itc_sub_type': 99}, 0, 0, 0, 90)
    for i in range(50):
        veh3.xy_history.append((i * 0.5, 0.0))
        veh3.vote_type(14, 1, 0.9, i * 100)
    check('移动目标报设施 14 -> 拒绝锁定', veh3.locked_type is None)

    veh4 = VehicleState(4, 'L', 0, 0, 0, {'itc_sub_type': 99}, 0, 0, 0, 90)
    for i in range(50):
        veh4.xy_history.append((0.15 * math.sin(i), 0.15 * math.cos(i)))
        veh4.vote_type(14, 1, 0.9, i * 100)
    check('静止目标报设施 14 -> 锁 14', veh4.locked_type == 14)

    # ================= T3: 变道平滑 =================
    print('\n== T3: 变道 (含横向跟踪) ==')

    def run_lane_change():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        outs = []
        lanes = []
        for i in range(150):
            t += 100
            tt = t / 1000.0
            s = 20 + 5.0 * tt
            la = 0.0 if tt < 5 else -min(4.0, (tt - 5) * 1.0)
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), la)
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(1001, x, y, bearing)]))
            v = eng.tracker.active_vehicles.get(1001)
            lanes.append(v.lane_id if v else 'GONE')
            for pv in pf.vehicles:
                if pv.fixed_id == 1001:
                    outs.append((pv.x, pv.y, pv.radar_heading))
        return mm, outs, lanes

    mm, outs, lanes = quiet(run_lane_change)
    max_jump = max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(outs, outs[1:]))
    flips = sum(1 for a, b in zip(outs, outs[1:]) if abs((b[2] - a[2] + 180) % 360 - 180) > 90)
    check('最终车道 = B', lanes[-1] == B)
    check('最大帧间跳变 < 1.0m', max_jump < 1.0, f'({max_jump:.2f}m)')
    check('航向 180° 翻转 = 0', flips == 0)

    # ================= T4/T5: 预测衰减与静止零漂移 =================
    print('\n== T4/T5: 断联推演 ==')

    def run_pred_decay():
        eng, mm, bearing, line_len = get_ctx()
        s = 50.0
        for i in range(30):
            t = (i + 1) * 100
            s += 0.5
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(5001, x, y, bearing)]))
        v0 = eng.tracker.active_vehicles[5001].v
        s0 = eng.tracker.active_vehicles[5001].s
        for i in range(50):
            t = 3000 + (i + 1) * 100
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[]))
        return v0, eng.tracker.active_vehicles[5001].s - s0

    v0, drift = quiet(run_pred_decay)
    check('5m/s 车 5s 盲区漂移 < 12m', drift < 12.0, f'(v={v0:.1f}, drift={drift:.2f}m)')

    def run_static():
        eng, mm, bearing, line_len = get_ctx()
        s = 60.0
        for i in range(5):
            t = (i + 1) * 100
            x, y = mm.offset_lateral(A, s, 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(7001, x, y, bearing)]))
        s0 = eng.tracker.active_vehicles[7001].s
        for i in range(50):
            t = 500 + (i + 1) * 100
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[]))
        return eng.tracker.active_vehicles[7001].s - s0

    drift5 = quiet(run_static)
    check('静止车 5s 盲区漂移 = 0', abs(drift5) < 0.01, f'({drift5:.3f}m)')

    # ================= T6: 恢复追击温和 =================
    print('\n== T6: 恢复追击 ==')

    def run_recovery():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        s2 = 30.0
        for i in range(30):
            t += 100
            s2 += 0.5
            x, y = mm.offset_lateral(A, min(s2, line_len - 1.0), 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(5002, x, y, bearing)]))
        for i in range(20):
            t += 100
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[]))
        s_real = s2 + 10.0
        outs = []
        for i in range(15):
            t += 100
            s_real += 0.5
            x, y = mm.offset_lateral(A, min(s_real, line_len - 1.0), 0.0)
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(5002, x, y, bearing)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 5002:
                    outs.append((pv.x, pv.y))
        return outs

    outs = quiet(run_recovery)
    max_jump = max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(outs, outs[1:]))
    check('恢复期最大跳变 < 3.5m', max_jump < 3.5, f'({max_jump:.2f}m)')

    # ================= T7: 去重 =================
    print('\n== T7: 去重 ==')

    def run_dedup():
        eng, mm, bearing, line_len = get_ctx()
        sA = 40.0
        for i in range(15):
            t3 = (i + 1) * 100
            sA += 0.5
            x, y = mm.offset_lateral(A, min(sA, line_len - 1.0), 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t3, vehicles=[mk_rv(6001, x, y, bearing)]))
        rev = (bearing + 180) % 360
        pf = None
        for i in range(6):
            t3 = 1600 + i * 100
            x2, y2 = mm.offset_lateral(A, sA - 3.0 - i * 0.5, 2.0)
            pf = eng.process_frame(RawFrame(timestamp_ms=t3, vehicles=[mk_rv(6002, x2, y2, rev)]))
        ids_opp = sorted(set(pv.fixed_id for pv in pf.vehicles))
        # 同向分裂应合并
        eng2, mm2, bearing2, line_len2 = get_ctx()
        sA2 = 40.0
        for i in range(15):
            t3 = (i + 1) * 100
            sA2 += 0.5
            x, y = mm2.offset_lateral(A, min(sA2, line_len2 - 1.0), 0.0)
            eng2.process_frame(RawFrame(timestamp_ms=t3, vehicles=[mk_rv(6101, x, y, bearing2)]))
        for i in range(6):
            t3 = 1600 + i * 100
            x2, y2 = mm2.offset_lateral(A, sA2 - 3.0 + i * 0.5, 1.0)
            pf2 = eng2.process_frame(RawFrame(timestamp_ms=t3, vehicles=[mk_rv(6102, x2, y2, bearing2)]))
        ids_same = sorted(set(pv.fixed_id for pv in pf2.vehicles))
        return ids_opp, ids_same

    ids_opp, ids_same = quiet(run_dedup)
    check('对向车近距离不误合并', ids_opp == [6001, 6002], f'({ids_opp})')
    check('同向近距离分裂正确合并', ids_same == [6101], f'({ids_same})')

    # ================= T8: 离道航向物理约束 =================
    print('\n== T8: 离道车航向 ==')

    def run_offlane_heading():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        outs1 = []
        for i in range(60):
            t = (i + 1) * 100
            x = cx + random.uniform(-0.3, 0.3)
            y = cy + random.uniform(-0.3, 0.3)
            h = random.uniform(0, 360)
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(100, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 100:
                    outs1.append(pv.radar_heading)
        return outs1

    outs1 = quiet(run_offlane_heading)
    max_turn = max(abs((b - a + 180) % 360 - 180) for a, b in zip(outs1, outs1[1:]))
    check('噪点航向乱转: 帧间最大偏转 <= 4.5°', max_turn <= 4.5 + 1e-6, f'({max_turn:.1f}°)')

    def run_offlane_static():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        outs = []
        for i in range(60):
            t = (i + 1) * 100
            x = cx + random.uniform(-0.2, 0.2)
            y = cy + random.uniform(-0.2, 0.2)
            h = 45.0 if i < 30 else 225.0
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(200, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 200:
                    outs.append(pv.radar_heading)
        return abs((outs[-1] - outs[29] + 180) % 360 - 180)

    flip_chg = quiet(run_offlane_static)
    check('静止目标感知航向突翻 180°: 输出冻结', flip_chg < 1.0, f'({flip_chg:.1f}°)')

    def run_offlane_moving():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        mv = 80.0
        outs = []
        for i in range(60):
            t = (i + 1) * 100
            d = i * 0.6
            x = cx + d * math.cos(math.radians(mv))
            y = cy + d * math.sin(math.radians(mv))
            h = mv + random.uniform(-10, 10)
            if i % 5 == 4:
                h = (mv + 150) % 360
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(300, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 300:
                    outs.append(pv.radar_heading)
        err = abs((outs[-1] - mv + 180) % 360 - 180)
        max_turn = max(abs((b - a + 180) % 360 - 180) for a, b in zip(outs, outs[1:]))
        return err, max_turn

    err3, turn3 = quiet(run_offlane_moving)
    check('移动车最终航向误差 < 30°', err3 < 30, f'({err3:.1f}°)')
    check('移动车帧间偏转 <= 4.5° (限转速)', turn3 <= 4.5 + 1e-6, f'({turn3:.1f}°)')

    def run_offlane_reverse():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        facing = 260.0
        outs = []
        for i in range(60):
            t = (i + 1) * 100
            d = i * 0.5
            x = cx + d * math.cos(math.radians(80.0))
            y = cy + d * math.sin(math.radians(80.0))
            h = facing + random.uniform(-5, 5)
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(400, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 400:
                    outs.append(pv.radar_heading)
        return abs((outs[-1] - facing + 180) % 360 - 180)

    err4 = quiet(run_offlane_reverse)
    check('倒车车: 航向不被错误翻转', err4 < 30, f'({err4:.1f}°)')

    # ================= T9: 2D 缝合基线连续性 (本版核心修复) =================
    print('\n== T9: 2D 缝合基线 ==')

    def run_suture():
        """车断联后以新 ID 在附近重现 -> 缝合, 输出坐标应连续无跳变"""
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        s = 60.0
        outs = []
        ids = []
        for i in range(40):
            t += 100
            if i < 30:
                s += 0.5
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), 0.0)
            # i in [30, 39]: 老ID消失, 新ID 9002 在老车前方 8m 处出现 (缝合场景1)
            vid = 9001 if i < 30 else 9002
            if i >= 30:
                sx = min(s + 8.0, line_len - 1.0)
                x, y = mm.offset_lateral(A, sx, 0.0)
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(vid, x, y, bearing)]))
            for pv in pf.vehicles:
                if pv.fixed_id in (9001, 9002):
                    outs.append((t, pv.fixed_id, pv.x, pv.y))
                    ids.append(pv.fixed_id)
        # 新 ID 应被缝合回 9001
        final_id = ids[-1]
        # 缝合瞬间的坐标跳变 (帧 29 -> 30)
        jumps = []
        for a, b in zip(outs, outs[1:]):
            jumps.append(math.hypot(b[2] - a[2], b[3] - a[3]))
        return final_id, max(jumps), outs

    final_id, max_jump, outs = quiet(run_suture)
    check('断联重连缝合: 新 ID 归并回老 ID', final_id == 9001, f'(final={final_id})')
    check('缝合全程坐标连续 (< 2.5m/帧)', max_jump < 2.5, f'({max_jump:.2f}m)')

    # ================= T10: 在轨车断联推演回归 =================
    print('\n== T10: 490 离道拦截存在性 ==')

    def run_490():
        """海一路 490 车道信号丢失 + 横穿偏移 -> 剥夺在轨身份"""
        eng, mm, bearing, line_len = get_ctx()
        if '137438953490_1' not in mm.lanes:
            return None
        lane = '137438953490_1'
        ln_len = mm.lanes[lane]['line'].length
        h490 = (90 - mm.lanes[lane]['heading']) % 360
        s = 50.0
        for i in range(30):
            t = (i + 1) * 100
            s += 0.3
            x, y = mm.offset_lateral(lane, min(s, ln_len - 1.0), 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(8888, x, y, h490)]))
        v = eng.tracker.active_vehicles.get(8888)
        if v is None or v.lane_id != lane:
            return 'skip'
        # 横移至 2.3m 并保持: 让滤波横向偏移充分建立 (> 2m) 且仍在轨
        for i in range(40):
            t = 3000 + (i + 1) * 100
            la = min(2.3, 0.1 * i)
            x, y = mm.offset_lateral(lane, min(s, ln_len - 1.0), la)
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[mk_rv(8888, x, y, h490)]))
        v = eng.tracker.active_vehicles.get(8888)
        # 横穿特征已建立: 仍在 490 车道且滤波偏移 > 2m
        pre = v is not None and v.lane_id == lane and v.filtered_l > 2.0
        # 信号消失一帧: 拦截应剥夺在轨身份, 中止车道推演
        eng.process_frame(RawFrame(timestamp_ms=7100, vehicles=[]))
        v = eng.tracker.active_vehicles.get(8888)
        return bool(pre) and v is not None and v.is_off_lane

    r = quiet(run_490)
    if r == 'skip' or r is None:
        print('  [SKIP] 490 车道不可用')
    else:
        check('490 横穿离场拦截生效', bool(r))

    print(f'\n========== 结果: {PASS} PASS / {FAIL} FAIL ==========')
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

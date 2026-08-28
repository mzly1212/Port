# -*- coding: utf-8 -*-
"""
regression_test.py — tracker_v2 回归测试
覆盖 v1/refactor 版全部历史优化用例, 验证 MTT 架构重写后行为不回退。运行:
    D:\\anaconda\\envs\\CAR\\python.exe regression_test.py
"""
import math
import random

import bootstrap  # noqa: F401  (根目录 -> sys.path)
from data import RawVehicle, RawFrame

from engine import PerceptionFilterEngine
from track import Track

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


A = '137439016690'
B = '137439016674'


def get_ctx():
    eng = PerceptionFilterEngine()
    mm = eng.lane_map
    bearing = mm.lanes[A]['heading_math']  # 数学角航向
    line_len = mm.lanes[A]['length']
    return eng, mm, bearing, line_len


def mk_rv(vid, x, y, heading, sub=11):
    return RawVehicle(object_id=vid, lon=0, lat=0, rel_x=x, rel_y=y,
                      itc_obj_type=1, itc_sub_type=sub, plate_num='',
                      lane_no='', radar_heading=heading, type_reliability=0.9)


def main():
    random.seed(7)

    # ================= T1: 方向滞回 + 倒车航向稳定 =================
    print('\n== T1: 方向滞回 (车头 180° 翻转 / 倒行) ==')

    def run_t1():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        s = 20.0
        outs = []
        dirs = []
        for i in range(200):
            t += 100
            s += 0.04 if i < 100 else -0.15
            noisy = s + 0.35 * math.sin(2 * math.pi * (t / 1000.0) / 2.5) \
                + random.uniform(-0.1, 0.1)
            x, y = mm.offset_lateral(A, min(max(noisy, 1.0), line_len - 1.0), 0.0)
            # 车头始终朝 bearing: 反向行驶 = 倒车, 车头不应翻转
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(1, x, y, bearing)]))
            tr = eng.tracker.tracks.get(1)
            dirs.append(tr.dir if tr else 'GONE')
            for pv in pf.vehicles:
                if pv.fixed_id == 1:
                    outs.append(pv.radar_heading)
        return eng, outs, dirs, bearing

    eng, outs, dirs, bearing = run_t1()
    dir_flips = sum(1 for a, b in zip(dirs, dirs[1:])
                    if a != b and a is not None and b is not None)
    head_flips = sum(1 for a, b in zip(outs, outs[1:])
                     if abs((b - a + 180) % 360 - 180) > 90)
    err = abs((outs[-1] - bearing + 180) % 360 - 180)
    check('噪声期方向翻转次数 <= 2', dir_flips <= 2, f'(flips={dir_flips})')
    check('最终方向 = -1 (反向行驶)', dirs[-1] == -1, f'(dir={dirs[-1]})')
    check('倒行时车头不翻转 (航向 180° 跳变 = 0)', head_flips == 0)
    check('最终航向仍朝原方向 (倒车语义)', err < 30, f'(err={err:.1f}°)')

    # ================= T2: 车型投票 =================
    print('\n== T2: 车型投票锁定 ==')
    tr2 = Track(2, mk_rv(2, 0, 0, 90, sub=99), 0)
    for i in range(3):
        tr2.vote_type(11, 0.9, i * 100)
    for i in range(3, 43):
        tr2.vote_type(6, 0.9, i * 100)
    check('早期误分类 3x11 后持续 6 -> 锁 6', tr2.locked_type == 6,
          f'(got {tr2.locked_type})')

    tr3 = Track(3, mk_rv(3, 0, 0, 90, sub=99), 0)
    for i in range(50):
        tr3.meas_history.append((i * 100, i * 0.5, 0.0))
        tr3.vote_type(14, 0.9, i * 100)
    check('移动目标报设施 14 -> 拒绝锁定', tr3.locked_type is None)

    tr4 = Track(4, mk_rv(4, 0, 0, 90, sub=99), 0)
    for i in range(50):
        tr4.meas_history.append((i * 100, 0.15 * math.sin(i), 0.15 * math.cos(i)))
        tr4.vote_type(14, 0.9, i * 100)
    check('静止目标报设施 14 -> 锁 14', tr4.locked_type == 14)

    # ================= T3: 变道平滑 =================
    print('\n== T3: 变道 (含横向跟踪) ==')

    def run_lane_change():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        outs = []
        for i in range(150):
            t += 100
            tt = t / 1000.0
            s = 20 + 5.0 * tt
            la = 0.0 if tt < 5 else -min(4.0, (tt - 5) * 1.0)
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), la)
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(1001, x, y, bearing)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 1001:
                    outs.append((pv.x, pv.y, pv.radar_heading))
        return eng, outs

    eng, outs = run_lane_change()
    max_jump = max(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(outs, outs[1:]))
    flips = sum(1 for a, b in zip(outs, outs[1:])
                if abs((b[2] - a[2] + 180) % 360 - 180) > 90)
    final_lane = eng.tracker.tracks[1001].lane_id
    check('最终车道 = B', final_lane == B, f'(got {final_lane})')
    check('最大帧间跳变 < 1.0m', max_jump < 1.0, f'({max_jump:.2f}m)')
    check('航向 180° 翻转 = 0', flips == 0)

    # ================= T4/T5: 断联推演 =================
    print('\n== T4/T5: 断联推演 (衰减/零漂移/预测标记) ==')

    def run_pred_decay():
        eng, mm, bearing, line_len = get_ctx()
        s = 50.0
        for i in range(30):
            t = (i + 1) * 100
            s += 0.5
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t,
                                       vehicles=[mk_rv(5001, x, y, bearing)]))
        p0 = eng.tracker.tracks[5001].kf.pos
        pred_flag = None
        for i in range(50):
            t = 3000 + (i + 1) * 100
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[]))
            if t == 4000:
                for pv in pf.vehicles:
                    if pv.fixed_id == 5001:
                        pred_flag = pv.is_predicted
        p1 = eng.tracker.tracks[5001].kf.pos
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]), pred_flag

    drift, pred_flag = run_pred_decay()
    check('5m/s 车 5s 盲区漂移 < 13m', drift < 13.0, f'(drift={drift:.2f}m)')
    check('盲区 >500ms 标记 is_predicted', pred_flag is True)

    def run_static():
        eng, mm, bearing, line_len = get_ctx()
        s = 60.0
        for i in range(30):
            t = (i + 1) * 100
            x, y = mm.offset_lateral(A, s, 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t,
                                       vehicles=[mk_rv(7001, x, y, bearing)]))
        p0 = eng.tracker.tracks[7001].kf.pos
        for i in range(50):
            t = 3000 + (i + 1) * 100
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[]))
        p1 = eng.tracker.tracks[7001].kf.pos
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    drift5 = run_static()
    check('静止车 5s 盲区漂移 < 0.05m', abs(drift5) < 0.05, f'({drift5:.3f}m)')

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
            eng.process_frame(RawFrame(timestamp_ms=t,
                                       vehicles=[mk_rv(5002, x, y, bearing)]))
        for i in range(20):
            t += 100
            eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[]))
        s_real = s2 + 10.0
        outs = []
        for i in range(15):
            t += 100
            s_real += 0.5
            x, y = mm.offset_lateral(A, min(s_real, line_len - 1.0), 0.0)
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(5002, x, y, bearing)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 5002:
                    outs.append((pv.x, pv.y))
        return outs

    outs = run_recovery()
    max_jump = max(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(outs, outs[1:]))
    check('恢复期最大跳变 < 3.5m', max_jump < 3.5, f'({max_jump:.2f}m)')

    # ================= T7: 分裂/对向 =================
    print('\n== T7: 分裂去重与对向车隔离 ==')

    def run_dedup():
        eng, mm, bearing, line_len = get_ctx()
        sA = 40.0
        for i in range(15):
            t3 = (i + 1) * 100
            sA += 0.5
            x, y = mm.offset_lateral(A, min(sA, line_len - 1.0), 0.0)
            eng.process_frame(RawFrame(timestamp_ms=t3,
                                       vehicles=[mk_rv(6001, x, y, bearing)]))
        rev = (bearing + 180) % 360
        pf = None
        for i in range(6):
            t3 = 1600 + i * 100
            x2, y2 = mm.offset_lateral(A, sA - 3.0 - i * 0.5, 2.0)
            pf = eng.process_frame(RawFrame(timestamp_ms=t3,
                                            vehicles=[mk_rv(6002, x2, y2, rev)]))
        ids_opp = sorted(set(pv.fixed_id for pv in pf.vehicles))

        # 同向分裂应合并
        eng2, mm2, bearing2, line_len2 = get_ctx()
        sA2 = 40.0
        for i in range(15):
            t3 = (i + 1) * 100
            sA2 += 0.5
            x, y = mm2.offset_lateral(A, min(sA2, line_len2 - 1.0), 0.0)
            eng2.process_frame(RawFrame(timestamp_ms=t3,
                                        vehicles=[mk_rv(6101, x, y, bearing2)]))
        for i in range(6):
            t3 = 1600 + i * 100
            x2, y2 = mm2.offset_lateral(A, sA2 - 3.0 + i * 0.5, 1.0)
            pf2 = eng2.process_frame(RawFrame(
                timestamp_ms=t3,
                vehicles=[mk_rv(6102, x2, y2, bearing2)]))
        ids_same = sorted(set(pv.fixed_id for pv in pf2.vehicles))
        return ids_opp, ids_same

    ids_opp, ids_same = run_dedup()
    check('对向车近距离不误合并', ids_opp == [6001, 6002], f'({ids_opp})')
    check('同向近距离分裂正确合并', ids_same == [6101], f'({ids_same})')

    # ================= T8: 离道航向物理约束 =================
    print('\n== T8: 离道车航向 ==')

    def run_offlane_noise():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        outs1 = []
        for i in range(60):
            t = (i + 1) * 100
            x = cx + random.uniform(-0.3, 0.3)
            y = cy + random.uniform(-0.3, 0.3)
            h = random.uniform(0, 360)
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(100, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 100:
                    outs1.append(pv.radar_heading)
        return outs1

    outs1 = run_offlane_noise()
    max_turn = max(abs((b - a + 180) % 360 - 180)
                   for a, b in zip(outs1, outs1[1:]))
    check('噪点航向乱转: 帧间最大偏转 <= 4.5°', max_turn <= 4.5 + 1e-6,
          f'({max_turn:.1f}°)')

    def run_offlane_static():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        outs = []
        for i in range(60):
            t = (i + 1) * 100
            x = cx + random.uniform(-0.2, 0.2)
            y = cy + random.uniform(-0.2, 0.2)
            h = 45.0 if i < 30 else 225.0
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(200, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 200:
                    outs.append(pv.radar_heading)
        return abs((outs[-1] - outs[20] + 180) % 360 - 180)

    flip_chg = run_offlane_static()
    check('静止目标感知航向突翻 180°: 输出冻结', flip_chg < 1.0,
          f'({flip_chg:.1f}°)')

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
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(300, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 300:
                    outs.append(pv.radar_heading)
        err = abs((outs[-1] - mv + 180) % 360 - 180)
        max_turn = max(abs((b - a + 180) % 360 - 180)
                       for a, b in zip(outs, outs[1:]))
        return err, max_turn

    err3, turn3 = run_offlane_moving()
    check('移动车最终航向误差 < 30°', err3 < 30, f'({err3:.1f}°)')
    check('移动车帧间偏转 <= 4.5° (限转速)', turn3 <= 4.5 + 1e-6,
          f'({turn3:.1f}°)')

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
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(400, x, y, h)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 400:
                    outs.append(pv.radar_heading)
        return abs((outs[-1] - facing + 180) % 360 - 180)

    err4 = run_offlane_reverse()
    check('倒车车: 航向不被错误翻转', err4 < 30, f'({err4:.1f}°)')

    # ================= T9: 断联重连缝合 (ID 延续 + 坐标连续) =================
    print('\n== T9: 2D 缝合 ==')

    def run_suture():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        s = 60.0
        outs = []
        ids = []
        for i in range(40):
            t += 100
            if i < 30:
                s += 0.5
            vid = 9001 if i < 30 else 9002
            if i >= 30:
                sx = min(s + 8.0, line_len - 1.0)
                x, y = mm.offset_lateral(A, sx, 0.0)
            else:
                x, y = mm.offset_lateral(A, min(s, line_len - 1.0), 0.0)
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(vid, x, y, bearing)]))
            for pv in pf.vehicles:
                if pv.fixed_id in (9001, 9002):
                    outs.append((t, pv.fixed_id, pv.x, pv.y))
                    ids.append(pv.fixed_id)
        return ids[-1], max(math.hypot(b[2] - a[2], b[3] - a[3])
                            for a, b in zip(outs, outs[1:]))

    final_id, max_jump = run_suture()
    check('断联重连缝合: 新 ID 归并回老 ID', final_id == 9001,
          f'(final={final_id})')
    check('缝合全程坐标连续 (< 2.5m/帧)', max_jump < 2.5, f'({max_jump:.2f}m)')

    # ================= T10: 设施锚定与误绑定门卫 =================
    print('\n== T10: 固定设施 ==')

    def run_facility_pin():
        eng, mm, bearing, line_len = get_ctx()
        cx, cy = mm.offset_lateral(A, 80.0, 30.0)
        outs = []
        for i in range(60):
            t = (i + 1) * 100
            x = cx + random.uniform(-0.2, 0.2)
            y = cy + random.uniform(-0.2, 0.2)
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(800, x, y, 0.0, sub=14)]))
            for pv in pf.vehicles:
                if pv.fixed_id == 800:
                    outs.append((pv.x, pv.y))
        tr = eng.tracker.tracks.get(800)
        return tr, outs[-40:]

    tr_fac, tail = run_facility_pin()
    mx = sum(p[0] for p in tail) / len(tail)
    my = sum(p[1] for p in tail) / len(tail)
    scatter = max(math.hypot(p[0] - mx, p[1] - my) for p in tail)
    check('静止设施: 锚定确认', tr_fac is not None and tr_fac.is_facility)
    check('静止设施: 输出位置钉死 (< 0.15m)', scatter < 0.15,
          f'(scatter={scatter:.3f}m)')

    def run_misbind():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        s = 40.0
        final_pvs = []
        for i in range(28):
            t += 100
            s += 0.5
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), 0.0)
            # 后 8 帧: 车辆 ID 突然被误报为设施(14)
            sub = 11 if i < 20 else 14
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(850, x, y, bearing, sub=sub)]))
            final_pvs = [pv for pv in pf.vehicles]
        return final_pvs

    pvs = run_misbind()
    ids = set(pv.fixed_id for pv in pvs)
    sub_ok = all(pv.itc_sub_type == 11 for pv in pvs if pv.fixed_id == 850)
    check('设施点误绑定: 不产生分裂对象', ids == {850}, f'({ids})')
    check('设施点误绑定: 车型不被污染', sub_ok)

    # ================= T11: ID 跳变 + 雷达前后混淆 =================
    print('\n== T11: ID 跳变下方向稳定 ==')

    def run_id_jump():
        eng, mm, bearing, line_len = get_ctx()
        t = 0
        s = 40.0
        outs = []
        for i in range(60):
            t += 100
            s += 0.5
            x, y = mm.offset_lateral(A, min(s, line_len - 1.0), 0.0)
            # 雷达前后混淆: 每 3 帧航向翻转 180°
            h = bearing if (i // 3) % 2 == 0 else (bearing + 180) % 360
            # 感知 ID 每 10 帧跳变
            vid = 100 + (i // 10)
            pf = eng.process_frame(RawFrame(timestamp_ms=t,
                                            vehicles=[mk_rv(vid, x, y, h)]))
            for pv in pf.vehicles:
                outs.append((pv.fixed_id, pv.radar_heading))
        return eng, outs, bearing

    eng, outs, bearing = run_id_jump()
    n_tracks = len(eng.tracker.tracks)
    max_turn = max(abs((b[1] - a[1] + 180) % 360 - 180)
                   for a, b in zip(outs, outs[1:]))
    err = abs((outs[-1][1] - bearing + 180) % 360 - 180)
    check('ID 高频跳变: 航迹唯一且 ID 延续',
          n_tracks == 1 and outs[-1][0] == 100,
          f'(tracks={n_tracks}, final={outs[-1][0]})')
    check('前后混淆: 输出航向帧间偏转 <= 4.5°', max_turn <= 4.5 + 1e-6,
          f'({max_turn:.1f}°)')
    check('前后混淆: 最终航向误差 < 30°', err < 30, f'({err:.1f}°)')

    # ================= T12: 静止车队不误合并 =================
    print('\n== T12: 静止车队 ==')

    def run_queue():
        eng, mm, bearing, line_len = get_ctx()
        for i in range(60):
            t = (i + 1) * 100
            x1, y1 = mm.offset_lateral(A, 50.0, 0.0)
            x2, y2 = mm.offset_lateral(A, 42.0, 0.3)
            pf = eng.process_frame(RawFrame(timestamp_ms=t, vehicles=[
                mk_rv(9101, x1, y1, bearing),
                mk_rv(9102, x2, y2, bearing)]))
        return sorted(pv.fixed_id for pv in pf.vehicles)

    ids = run_queue()
    check('静止车队 8m 间距: 不误合并', ids == [9101, 9102], f'({ids})')

    # ==========================================================
    print(f'\n========== 结果: {PASS} PASS / {FAIL} FAIL ==========')
    return FAIL


if __name__ == '__main__':
    import sys
    sys.exit(1 if main() else 0)

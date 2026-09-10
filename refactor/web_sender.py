# -*- coding: utf-8 -*-
"""
web_sender.py — 多前端 TCP 推送

功能:
  - 支持同时向多个前端 (ip, port) 推送处理后的数据
  - 每个目标独立连接、独立断线重连, 单个前端故障不影响其他目标
  - 发送带超时: 防「半死前端」(TCP 层活着但应用不收数据) sendall
    无限阻塞冻结主循环 -> UDP 缓冲撑爆静默丢包 (生产事故教训)
  - 启动时每个目标只做一次连接尝试, 失败者进入周期重连 ——
    多目标下一个永久不可达的前端不能阻塞程序启动

配置 (config.py):
    # 处理后的数据推送给列表中的所有前端, 每个目标独立重连
    WEB_TARGETS = [
        ("10.28.49.196", 7098),
        ("192.168.1.100", 7098),   # 新增前端: 在此追加 (ip, port)
    ]
    WEB_SEND_TIMEOUT = 2.0         # 发送/连接超时(秒)
    WEB_RECONNECT_INTERVAL = 5.0   # 断线重连尝试间隔(秒)
"""
import socket
import time


class _DummyLogger:
    """未传入 logger 时的兜底输出"""
    def info(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)


class WebSender:
    """多前端 TCP 推送器: 每个目标一条独立连接"""

    def __init__(self, config, logger=None):
        self.timeout = getattr(config, 'WEB_SEND_TIMEOUT', 2.0)
        self.retry_interval = getattr(config, 'WEB_RECONNECT_INTERVAL', 5.0)
        self.logger = logger or _DummyLogger()

        targets = getattr(config, 'WEB_TARGETS', None)
        if not targets:
            print('error: no web targets')
        self.targets = [(str(ip), int(port)) for ip, port in targets]
        self.socks = [None] * len(self.targets)
        self.last_attempt = [0.0] * len(self.targets)

        # 启动: 每个目标单次尝试, 失败者进入周期重连模式
        for i in range(len(self.targets)):
            self._try_connect(i, startup=True)

    # ------------------------------------------------------------------
    def _try_connect(self, i, startup=False):
        """尝试连接第 i 个目标 (带超时单次尝试, 不阻塞事件循环)"""
        ip, port = self.targets[i]
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((ip, port))
            self.socks[i] = s
            self.logger.info(f"🖥️ 前端 TCP 连接成功: {ip}:{port}")
            return True
        except OSError as e:
            self.socks[i] = None
            self.last_attempt[i] = time.time()
            tag = "初始连接" if startup else "重连"
            self.logger.error(f"前端 TCP {tag}失败 ({e}), 每 {self.retry_interval:.0f} "
                              f"秒重试: {ip}:{port}")
            return False

    # ------------------------------------------------------------------
    def send_all(self, tcp_bytes):
        """向全部目标推送同一帧数据; 返回成功发送的目标数"""
        ok_count = 0
        for i in range(len(self.targets)):
            if self._send_one(i, tcp_bytes):
                ok_count += 1
        return ok_count

    def _send_one(self, i, tcp_bytes):
        """向第 i 个目标发送, 断线时周期性重连"""
        ip, port = self.targets[i]
        if self.socks[i] is None:
            now = time.time()
            if now - self.last_attempt[i] < self.retry_interval:
                return False   # 距上次重试不足间隔, 本帧跳过该目标
            if not self._try_connect(i):
                return False
        try:
            self.socks[i].sendall(tcp_bytes)
            return True
        except OSError as e:
            # Broken pipe / Connection reset / 发送超时: 连接已不可信。
            # 超时时可能已写入半帧, 该 TCP 流不可复用, 必须关闭重建
            if isinstance(e, socket.timeout):
                reason = "发送超时 (前端疑似卡死未收数据)"
            else:
                reason = str(e)
            self.logger.error(f"前端 TCP 发送失败 ({reason}), 进入重连模式: {ip}:{port}")
            try:
                self.socks[i].close()
            except OSError:
                pass
            self.socks[i] = None
            return False

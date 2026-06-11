"""UDP 通信客户端，封装与 Lite3 运动主机的指令发送"""
import socket
import struct


class RobotUDPClient:
    """UDP 通信基类，封装与 Lite3 运动主机的指令发送"""

    def __init__(self, local_port=43897, ctrl_ip="192.168.2.1", ctrl_port=43893):
        self.local_port = local_port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('0.0.0.0', local_port))  # 必须绑定, 机器人通过此端口回复
        self.ctrl_addr = (ctrl_ip, ctrl_port)

    def send_command(self, code, param1=0, param2=0):
        print(f"发送命令：Code={code}, Param1={param1}, Param2={param2}")
        self._send_simple(code, param1, param2)

    def _send_simple(self, code, param1=0, param2=0):
        try:
            payload = struct.pack('<3i', code, param1, param2)
            self.server.sendto(payload, self.ctrl_addr)
        except Exception as e:
            print(f"发送命令时出错：{e}")

    def close(self):
        self.server.close()

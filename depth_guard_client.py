# depth_guard_client.py
import requests
from typing import Optional, Dict, Any


class DepthGuardClient:
    """
    机器狗深度相机安全检测客户端。

    用法：
        client = DepthGuardClient("http://192.168.2.1:8000")
        distance = client.get_front_distance()
        safe = client.is_front_safe(threshold=0.8)
    """

    def __init__(self, base_url: str = "http://192.168.2.1:8000", timeout: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_status(self) -> Dict[str, Any]:
        """
        获取狗端深度检测完整状态。

        返回示例：
        {
            "safe": true,
            "distance": 1.74,
            "valid_pixels": 270,
            "reason": "clear",
            "age": 0.01
        }
        """
        url = f"{self.base_url}/status"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            return {
                "safe": False,
                "distance": None,
                "valid_pixels": 0,
                "reason": f"request_failed: {e}",
                "age": None,
            }

    def get_front_distance(self) -> Optional[float]:
        """
        返回前方距离，单位 m。
        如果没有有效数据，返回 None。
        """
        status = self.get_status()
        distance = status.get("distance")

        if distance is None:
            return None

        try:
            return float(distance)
        except (TypeError, ValueError):
            return None

    def is_front_safe(self, threshold: float = 0.8) -> bool:
        """
        根据输入阈值判断前方是否安全。

        threshold:
            安全距离阈值，单位 m。
            例如 threshold=0.8 表示前方最近距离大于 0.8m 才安全。

        返回：
            True: 安全
            False: 不安全或无法获取数据
        """
        distance = self.get_front_distance()

        if distance is None:
            return False

        return distance > threshold


_default_client = DepthGuardClient()


def get_front_distance() -> Optional[float]:
    """
    简单函数版：返回前方距离，单位 m。
    """
    return _default_client.get_front_distance()


def is_front_safe(threshold: float = 0.8) -> bool:
    """
    简单函数版：根据阈值返回安全判断。
    """
    return _default_client.is_front_safe(threshold)
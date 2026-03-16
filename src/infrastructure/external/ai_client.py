"""
AI 客户端封装
提供统一的 AI 调用接口
"""
import os
import json
import base64
from typing import Dict, List, Optional
from dotenv import load_dotenv
import httpx
from src.ai_message_builder import (
    build_analysis_text_prompt,
    build_user_message_content,
)
from src.infrastructure.config.settings import AISettings
from src.infrastructure.config.env_manager import env_manager


class AIClient:
    """AI 客户端封装"""

    def __init__(self):
        self.settings: Optional[AISettings] = None
        self.base_url: str = ""
        self.refresh()

    def _load_settings(self) -> None:
        load_dotenv(dotenv_path=env_manager.env_file, override=True)
        self.settings = AISettings()

    def refresh(self) -> None:
        self._load_settings()
        self.base_url = self._initialize_base_url()

    def _initialize_base_url(self) -> str:
        """初始化 AI 请求基础地址"""
        if not self.settings or not self.settings.is_configured():
            print("警告：AI 配置不完整，AI 功能将不可用")
            return ""
        return (self.settings.base_url or "").strip().rstrip("/")

    def is_available(self) -> bool:
        """检查 AI 客户端是否可用"""
        return bool(self.settings and self.settings.is_configured() and self.base_url)

    @staticmethod
    def encode_image(image_path: str) -> Optional[str]:
        """将图片编码为 Base64"""
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"编码图片失败: {e}")
            return None

    async def analyze(
        self,
        product_data: Dict,
        image_paths: List[str],
        prompt_text: str
    ) -> Optional[Dict]:
        """
        分析商品数据

        Args:
            product_data: 商品数据
            image_paths: 图片路径列表
            prompt_text: 分析提示词

        Returns:
            分析结果
        """
        if not self.is_available():
            print("AI 客户端不可用")
            return None

        try:
            messages = self._build_messages(product_data, image_paths, prompt_text)
            response = await self._call_ai(messages)
            return self._parse_response(response)
        except Exception as e:
            print(f"AI 分析失败: {e}")
            return None

    def _build_messages(self, product_data: Dict, image_paths: List[str], prompt_text: str) -> List[Dict]:
        """构建 AI 消息"""
        product_json = json.dumps(product_data, ensure_ascii=False, indent=2)
        image_data_urls: List[str] = []
        for path in image_paths:
            base64_img = self.encode_image(path)
            if base64_img:
                image_data_urls.append(f"data:image/jpeg;base64,{base64_img}")

        text_prompt = build_analysis_text_prompt(
            product_json,
            prompt_text,
            include_images=bool(image_data_urls),
        )
        user_content = build_user_message_content(text_prompt, image_data_urls)
        return [{"role": "user", "content": user_content}]

    async def _call_ai(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """调用 AI API"""
        request_params = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # 根据配置添加可选参数
        if self.settings.enable_response_format:
            request_params["response_format"] = {"type": "json_object"}

        if self.settings.enable_thinking:
            request_params["enable_thinking"] = False

        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        proxy = (self.settings.proxy_url or "").strip() or None
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(
            timeout=360.0,
            proxy=proxy,
            trust_env=False,
            http2=False,
        ) as client:
            resp = await client.post(url, headers=headers, json=request_params)
            resp.raise_for_status()
            result = resp.json()
            return (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """解析 AI 响应"""
        try:
            # 直接解析 JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 清理 Markdown 代码块标记
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # 提取 JSON 对象
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = cleaned[start:end + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            print(f"无法解析 AI 响应: {response_text[:100]}")
            return None

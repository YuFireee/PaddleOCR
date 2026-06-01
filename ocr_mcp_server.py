#!/usr/bin/env python3
"""
PaddleOCR MCP Server (官方 API 对齐版)
支持本地文件和剪贴板图片识别
"""
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
from fastmcp import FastMCP

# ========== 请替换成你自己的隐私配置 ==========
JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
TOKEN = "ae23c511a7004dca8acac6bf83ca733e9077a855"
MODEL = "PaddleOCR-VL-1.6"
# ===========================================

mcp = FastMCP("PaddleOCR")

# 通用请求头
HEADERS = {
    "Authorization": f"bearer {TOKEN}",
}

# 可选参数（与官方一致）
OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}


def submit_job_file(file_path: str) -> str:
    """
    使用本地文件提交 OCR 任务，返回 job_id
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = {
        "model": MODEL,
        "optionalPayload": json.dumps(OPTIONAL_PAYLOAD)
    }

    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(JOB_URL, headers=HEADERS, data=data, files=files)

    if resp.status_code != 200:
        raise RuntimeError(f"提交任务失败 ({resp.status_code}): {resp.text}")

    job_id = resp.json()["data"]["jobId"]
    return job_id


def submit_job_url(url: str) -> str:
    """
    使用文件 URL 提交 OCR 任务（备用）
    """
    headers = {
        "Authorization": f"bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "fileUrl": url,
        "model": MODEL,
        "optionalPayload": OPTIONAL_PAYLOAD,
    }
    resp = requests.post(JOB_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"提交任务失败 ({resp.status_code}): {resp.text}")
    job_id = resp.json()["data"]["jobId"]
    return job_id


def poll_job_result(job_id: str) -> str:
    """
    轮询任务结果，返回提取的文本
    """
    while True:
        resp = requests.get(f"{JOB_URL}/{job_id}", headers=HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(f"查询任务状态失败 ({resp.status_code})")
        data = resp.json()["data"]
        state = data["state"]

        if state == "done":
            # 获取 JSONL 结果文件
            jsonl_url = data["resultUrl"]["jsonUrl"]
            jsonl_resp = requests.get(jsonl_url)
            jsonl_resp.raise_for_status()

            texts = []
            lines = jsonl_resp.text.strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                result = json.loads(line)["result"]
                for layout in result["layoutParsingResults"]:
                    md_text = layout["markdown"]["text"]
                    texts.append(md_text)

            return "\n".join(texts) if texts else "（未识别到任何文字）"

        elif state == "failed":
            error_msg = data.get("errorMsg", "未知错误")
            raise RuntimeError(f"识别失败: {error_msg}")

        else:
            # pending 或 running，等待 3 秒
            time.sleep(3)


# ==================== MCP 工具 ====================

@mcp.tool()
def ocr_file(image_path: str) -> str:
    """
    识别本地图片文件中的文字
    参数:
        image_path: 图片文件的路径，支持绝对路径或相对路径
    返回:
        识别出的文字内容
    """
    image_path = os.path.expanduser(image_path)
    job_id = submit_job_file(image_path)
    return poll_job_result(job_id)


@mcp.tool()
def ocr_clipboard() -> str:
    """
    识别系统剪贴板中的图片（如果剪贴板当前有图片）
    需要安装 Pillow 库（pip install Pillow）
    返回:
        识别出的文字内容
    """
    try:
        from PIL import ImageGrab
    except ImportError:
        return "错误：请先安装 Pillow 库 (pip install Pillow) 才能使用剪贴板功能"

    img = ImageGrab.grabclipboard()
    if img is None:
        return "剪贴板中没有图片数据"

    if isinstance(img, list):
        # 文件路径列表
        if not img:
            return "剪贴板为空"
        image_path = img[0]
        return ocr_file(image_path)

    else:
        # PIL Image 对象，保存为临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name
        try:
            return ocr_file(tmp_path)
        finally:
            os.unlink(tmp_path)


# ==================== 启动服务器 ====================
if __name__ == "__main__":
    mcp.run(transport="stdio")
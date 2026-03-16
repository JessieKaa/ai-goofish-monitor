import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from src.config import AI_API_READY, MODEL_NAME, ai_chat_completions, get_ai_request_params
from src.scraper import HookDecision, HookStage
from src.utils import log_time


_PROMPT_CACHE: dict[str, str] = {}


def _hook_log(message: str) -> None:
    log_time(f"[Hook] {message}")


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_text_file(path_value: str) -> str:
    path = Path(path_value)
    cache_key = str(path.resolve()) if path.exists() else path_value
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    content = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[cache_key] = content
    return content


def _load_pre_seller_prompt(task_config: dict, params: dict) -> str:
    # 优先从 hook_params 读取，其次从 task 顶层字段读取
    single_file = (
        params.get("pre_seller_ai_prompt_file")
        or task_config.get("pre_seller_ai_prompt_file")
    )
    if single_file:
        return _read_text_file(str(single_file))

    base_file = (
        params.get("pre_seller_ai_prompt_base_file")
        or task_config.get("pre_seller_ai_prompt_base_file")
    )
    criteria_file = (
        params.get("pre_seller_ai_prompt_criteria_file")
        or task_config.get("pre_seller_ai_prompt_criteria_file")
    )
    if base_file and criteria_file:
        base_prompt = _read_text_file(str(base_file))
        criteria_prompt = _read_text_file(str(criteria_file))
        return base_prompt.replace("{{CRITERIA_SECTION}}", criteria_prompt)

    raise ValueError(
        "缺少预判提示词配置。请设置 pre_seller_ai_prompt_file，"
        "或 pre_seller_ai_prompt_base_file + pre_seller_ai_prompt_criteria_file。"
    )


def _extract_json_block(text: str) -> dict:
    content = (text or "").strip()
    if not content:
        raise ValueError("AI 返回为空")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("未在 AI 返回中找到 JSON 对象")
    return json.loads(content[start_idx : end_idx + 1])


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value.strip()
            continue
        return value
    return ""


def _build_pre_seller_ai_input(
    *,
    task_config: dict,
    item_data: dict,
    seller_do: dict,
    detail_json: dict,
) -> dict:
    detail_data = (detail_json.get("data") or {})
    item_do = detail_data.get("itemDO") or {}
    image_infos = item_do.get("imageInfos") or []
    video_infos = item_do.get("videoInfos") or []

    summary = {
        "task_name": str(task_config.get("task_name", "")),
        "keyword": str(task_config.get("keyword", "")),
        "item_core": {
            "item_id": str(item_data.get("商品ID", "")),
            "title": str(_first_non_empty(item_data.get("商品标题"), item_do.get("title"))),
            "price_text": str(item_data.get("当前售价", "")),
            "original_price_text": str(item_data.get("商品原价", "")),
            "want_count": _first_non_empty(
                item_data.get("“想要”人数"), item_do.get("wantCnt")
            ),
            "browse_count": _first_non_empty(item_data.get("浏览量"), item_do.get("browseCnt")),
            "tags": (item_data.get("商品标签") or [])[:10],
            "area": str(item_data.get("发货地区", "")),
            "publish_time": str(item_data.get("发布时间", "")),
        },
        "detail_core": {
            "desc": str(_first_non_empty(item_do.get("desc"), item_do.get("title"))),
            "category": _first_non_empty(
                item_do.get("cateName"), item_do.get("categoryName"), item_do.get("spBizType")
            ),
            "image_count": len(image_infos),
            "has_video": bool(video_infos),
            "post_fee": _first_non_empty(item_do.get("postFee"), item_do.get("freight")),
            "city": _first_non_empty(item_do.get("city"), item_do.get("area")),
        },
        "seller_core": {
            "seller_id": seller_do.get("sellerId"),
            "nickname": str(_first_non_empty(seller_do.get("nick"), item_data.get("卖家昵称"))),
            "user_reg_day": seller_do.get("userRegDay"),
            "zhima_level": ((seller_do.get("zhimaLevelInfo") or {}).get("levelName")),
            "fans_count": seller_do.get("fansCount"),
            "follow_count": seller_do.get("followCount"),
            "sell_item_count": seller_do.get("itemCount"),
        },
    }

    return summary


def _write_pre_seller_ai_gate_log(
    *,
    task_name: str,
    item_id: str,
    keyword: str,
    payload: dict,
    gate_prompt: str,
    user_prompt: str,
    raw_response: str = "",
    parsed_response: dict | None = None,
    final_action: str = "",
    final_reason: str = "",
    error: str = "",
) -> None:
    try:
        logs_dir = os.path.join("logs", "ai")
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_task = "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in (task_name or "unknown")
        ).strip("_") or "unknown"
        filename = f"{ts}_pre_seller_gate_{safe_task}_{item_id or 'NA'}.log"
        filepath = os.path.join(logs_dir, filename)

        content = {
            "timestamp": ts,
            "hook": "ai_gate_before_seller_profile",
            "task_name": task_name,
            "keyword": keyword,
            "item_id": item_id,
            "request_payload": payload,
            "prompt_length": len(gate_prompt or ""),
            "prompt_text": gate_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_response": parsed_response or {},
            "final_action": final_action,
            "final_reason": final_reason,
            "error": error,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, ensure_ascii=False, indent=2))
        _hook_log(f"AI Gate 详细日志已写入: {filepath}")
    except Exception as e:
        _hook_log(f"写入 AI Gate 日志失败: {e}")


async def ai_gate_before_seller_profile(ctx: dict):
    """
    在采集卖家主页信息前调用 OpenAI 做预判，决定是否继续下一步。

    期望模型返回 JSON（字段不区分多余字段）:
    {
      "should_continue": true,
      "action": "continue|skip_item|skip_seller_profile|stop_task",
      "reason": "..."
    }
    """
    task_config = ctx.get("task_config") or {}
    params = task_config.get("hook_params") or {}

    fail_open = _to_bool(params.get("pre_seller_ai_fail_open"), True)
    default_action = str(params.get("pre_seller_ai_default_action", "skip_item"))
    task_name = str(task_config.get("task_name", "unknown"))
    keyword = str(task_config.get("keyword", ""))
    item_data = ctx.get("item_data") or {}
    item_id = str(item_data.get("商品ID", "N/A"))
    _hook_log(
        f"AI Gate 开始: task={task_name}, item_id={item_id}, fail_open={fail_open}, default_action={default_action}"
    )

    if not AI_API_READY:
        msg = "AI API 未就绪，跳过 pre-seller AI gate"
        _hook_log(msg)
        _write_pre_seller_ai_gate_log(
            task_name=task_name,
            item_id=item_id,
            keyword=keyword,
            payload={},
            gate_prompt="",
            user_prompt="",
            final_action="continue" if fail_open else "skip_item",
            final_reason=msg,
            error=msg,
        )
        return True if fail_open else HookDecision.skip_current_item(msg)

    try:
        gate_prompt = _load_pre_seller_prompt(task_config, params)
    except Exception as e:
        msg = f"读取 pre-seller prompt 失败: {e}"
        _hook_log(msg)
        _write_pre_seller_ai_gate_log(
            task_name=task_name,
            item_id=item_id,
            keyword=keyword,
            payload={},
            gate_prompt="",
            user_prompt="",
            final_action="continue" if fail_open else "skip_item",
            final_reason=msg,
            error=traceback.format_exc(),
        )
        return True if fail_open else HookDecision.skip_current_item(msg)

    seller_do = ctx.get("seller_do") or {}
    detail_json = ctx.get("detail_json") or {}
    llm_input = _build_pre_seller_ai_input(
        task_config=task_config,
        item_data=item_data,
        seller_do=seller_do,
        detail_json=detail_json,
    )

    user_prompt = (
        f"{gate_prompt}\n\n"
        "请基于下面的商品信息做决策，并只输出 JSON：\n"
        f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    _hook_log(
        f"AI Gate 请求准备完成: task={task_name}, item_id={item_id}, prompt_len={len(gate_prompt)}, input_len={len(user_prompt)}"
    )

    raw_content = ""
    parsed = None
    try:
        payload = get_ai_request_params(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        response = await ai_chat_completions(payload)
        raw_content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        parsed = _extract_json_block(raw_content)
    except Exception as e:
        msg = f"pre-seller AI gate 调用失败: {e}"
        _hook_log(msg)
        _write_pre_seller_ai_gate_log(
            task_name=task_name,
            item_id=item_id,
            keyword=keyword,
            payload=llm_input,
            gate_prompt=gate_prompt,
            user_prompt=user_prompt,
            raw_response=raw_content,
            parsed_response=parsed,
            final_action="continue" if fail_open else "skip_item",
            final_reason=msg,
            error=traceback.format_exc(),
        )
        if fail_open:
            return True
        return HookDecision.skip_current_item(msg)

    should_continue = bool(parsed.get("should_continue", True))
    action = str(parsed.get("action", "")).strip().lower()
    reason = str(parsed.get("reason", "")).strip()

    if not action:
        if should_continue:
            action = "continue"
        else:
            action = str(default_action).strip().lower() or "skip_item"

    _hook_log(
        f"AI Gate 决策: task={task_name}, item_id={item_id}, action={action}, reason={reason or 'N/A'}"
    )
    _write_pre_seller_ai_gate_log(
        task_name=task_name,
        item_id=item_id,
        keyword=keyword,
        payload=llm_input,
        gate_prompt=gate_prompt,
        user_prompt=user_prompt,
        raw_response=raw_content,
        parsed_response=parsed,
        final_action=action,
        final_reason=reason,
    )

    if action == "continue":
        return {
            "proceed": True,
            "reason": reason or "AI gate allow",
            "updates": {"pre_seller_ai_gate": parsed},
        }
    if action == "skip_seller_profile":
        return {
            "proceed": True,
            "reason": reason or "AI gate skip seller profile",
            "updates": {
                "skip_seller_profile": True,
                "pre_seller_ai_gate": parsed,
            },
        }
    if action == "stop_task":
        return HookDecision.stop_whole_task(reason or "AI gate stop task")

    # 默认按 skip_item 处理
    return HookDecision.skip_current_item(reason or "AI gate skip item")


def skip_low_quality_item(ctx: dict):
    """
    示例：在 ITEM_READY_FOR_ANALYSIS 阶段，基于商品详情决定是否继续 AI 分析流程。
    可通过 task_config['hook_params'] 自定义阈值。
    """
    task_config = ctx.get("task_config") or {}
    params = task_config.get("hook_params") or {}
    min_want_count = int(params.get("min_want_count", 3))
    max_registration_days = int(params.get("max_registration_days", 7))

    item_data = ctx.get("item_data") or {}
    final_record = ctx.get("final_record") or {}
    seller_info = final_record.get("卖家信息") or {}

    want_cnt = item_data.get("“想要”人数")
    try:
        want_cnt_int = int(want_cnt)
    except (TypeError, ValueError):
        want_cnt_int = 0

    # 卖家注册天数在原始详情里更稳定，优先读 detail_json
    detail_json = ctx.get("detail_json") or {}
    seller_do = ((detail_json.get("data") or {}).get("sellerDO") or {})
    reg_days_raw = seller_do.get("userRegDay", 0)
    try:
        reg_days = int(reg_days_raw)
    except (TypeError, ValueError):
        reg_days = 0

    if want_cnt_int < min_want_count:
        return HookDecision.skip_current_item(
            f"想要人数过低({want_cnt_int}<{min_want_count})，跳过后续分析"
        )

    if 0 < reg_days < max_registration_days:
        return HookDecision.skip_current_item(
            f"卖家注册时长过短({reg_days}天<{max_registration_days}天)，跳过后续分析"
        )

    # 你也可以在这里修改记录内容，供后续流程使用
    seller_info["hook_checked"] = True
    final_record["卖家信息"] = seller_info
    return HookDecision.continue_next()


def block_notification_for_blacklist_words(ctx: dict):
    """示例：在 BEFORE_NOTIFICATION 阶段拦截通知发送。"""
    item = ctx.get("item_data") or {}
    title = str(item.get("商品标题", ""))
    blocked_words = {"高仿", "代购证书"}
    if any(word in title for word in blocked_words):
        return {
            "proceed": False,
            "skip_item": True,
            "reason": f"命中通知黑名单词，拦截通知: {title}",
        }
    return True


def skip_seller_profile_for_risk_item(ctx: dict):
    """
    示例：在 BEFORE_SELLER_PROFILE 阶段，只跳过卖家主页采集，不跳过整件商品流程。
    """
    item = ctx.get("item_data") or {}
    title = str(item.get("商品标题", ""))
    risky_words = {"全新未拆", "急出"}
    if any(word in title for word in risky_words):
        return {
            "proceed": True,
            "reason": f"标题命中快速规则，跳过卖家主页采集: {title}",
            "updates": {"skip_seller_profile": True},
        }
    return True


HOOKS_EXAMPLE = {
    HookStage.BEFORE_SELLER_PROFILE: [
        "src.hooks.item_hooks:ai_gate_before_seller_profile",
        "src.hooks.item_hooks:skip_seller_profile_for_risk_item",
    ],
    HookStage.ITEM_READY_FOR_ANALYSIS: [
        "src.hooks.item_hooks:skip_low_quality_item",
    ],
    HookStage.BEFORE_NOTIFICATION: [
        "src.hooks.item_hooks:block_notification_for_blacklist_words",
    ],
}

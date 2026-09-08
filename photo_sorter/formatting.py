from __future__ import annotations


def format_elapsed_time(seconds: float) -> str:
    """Format total wall time, rather than adding parallel per-image durations."""
    total_seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分{seconds_part:02d}秒"
    if minutes:
        return f"{minutes}分{seconds_part:02d}秒"
    return f"{seconds_part}秒"

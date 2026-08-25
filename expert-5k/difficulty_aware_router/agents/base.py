from __future__ import annotations

import re
import warnings
from pathlib import Path

from cache import CacheOpenAI

SUMMARY_PATTERN = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL)


class BaseAgent:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        cache_mode: str = "record",
    ) -> None:
        self.client = CacheOpenAI(
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
        )

    def extract_chat_text(self, response: object) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""

    def extract_summary(self, text: str) -> str:
        """Extract the longest non-empty summary block from model output.

        The parser first looks for well-formed ``<summary>...</summary>`` blocks.
        If none are found, it applies a narrow fallback for the known formatting
        failure mode where the model emits ``<summary>`` but forgets the closing
        ``</summary>`` tag. In that case, it appends the missing closing tag to
        the end of the text and retries once. All other malformed formats are
        left as failures so the caller can treat them as invalid output.
        """
        summaries = [match.strip() for match in SUMMARY_PATTERN.findall(text)]

        if not summaries and "<summary>" in text and "</summary>" not in text:
            repaired_text = text.rstrip() + "\n</summary>"
            summaries = [
                match.strip() for match in SUMMARY_PATTERN.findall(repaired_text)
            ]

        if not summaries:
            warnings.warn(
                "Missing <summary>...</summary> block in summary text.",
                stacklevel=2,
            )
            return ""

        non_empty_summaries = [summary for summary in summaries if summary]
        if not non_empty_summaries:
            warnings.warn(
                "Extracted summary block is empty.",
                stacklevel=2,
            )
            return ""

        return max(non_empty_summaries, key=len)

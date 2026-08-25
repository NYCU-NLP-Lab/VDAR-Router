from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from .base import BaseAgent
from .settings import get_analysis_settings

V1_SYSTEM_PROMPT = dedent(
    """
    Your role as an assistant is to analyze the difficulty of a given query for a large language model
    through a systematic long thinking process analysis. You will be provided with the user query.
    You need to evaluate the incoming query on several key dimensions: reasoning, comprehension,
    instruction following, agentic, knowledge retrieval, coding, multilingual. For each dimension,
    elaborate on the specific challenges and required capabilities. Please structure your response into Summary.
    In the Summary section, based on the analysis, explorations, and reflections from the Think section, systematically present the summary you think for the query difficulty.
    The summary should remain a clear, concise expression style and detail necessary difficulty description to reach the conclusion, formatted as follows: <summary> {final formatted, precise, and clear summary} </summary>.
    Now, try to analyze the following query through the above guidelines:
    """
).strip()

V2_SYSTEM_PROMPT = dedent(
    """
    You are given only a user query, without any reference answer or model response. Your task is to analyze the query’s difficulty profile for a large language model.

    Think carefully about the query before answering, but do not output your reasoning process.

    Important:
    - Analyze the difficulty of answering this query based only on the query itself.
    - Do not assume a specific answer.
    - Do not evaluate answer quality.
    - Focus on what capabilities and challenges the query would impose on an LLM.

    Consider the following candidate difficulty dimensions:
    - reasoning
    - comprehension
    - instruction_following
    - agentic
    - knowledge_retrieval
    - coding
    - multilingual

    Rules:
    1. Only include dimensions that are genuinely relevant to the query.
    2. Do not force all dimensions into the answer.
    3. Select at most 3 primary dimensions.
    4. Prioritize the dimensions that dominate the query's difficulty, not merely those that are present.
    5. Focus on concrete sources of difficulty, such as ambiguity, multi-step reasoning, hidden constraints, external knowledge dependence, action planning, code synthesis/debugging, or language mixing.
    6. Do not use generic statements unless they are tied to this specific query.
    7. Keep the output compact, information-dense, and suitable for downstream similarity search, retrieval, embedding, or router feature extraction.
    8. Keep difficulty_profile within 1–3 sentences.

    Output exactly in this format:

    <summary>
    overall_difficulty: [low|medium|high]
    primary_dimensions: [dimension_1, dimension_2, ...]
    difficulty_profile: [A compact description of the main challenges this query would pose to an LLM, based only on the query itself.]
    </summary>
    """
).strip()


PROMPT_REGISTRY: dict[str, str] = {"v1": V1_SYSTEM_PROMPT, "v2": V2_SYSTEM_PROMPT}

DIFFICULTY_ANALYSIS_USER_PROMPT = dedent(
    """
    **Query to Analyze:**
    {question}
    """
).strip()


@dataclass(frozen=True)
class DifficultyAnalysisResult:
    response_text: str
    summary: str
    input_token_count: int | None = None


class DifficultyAnalysisAgent(BaseAgent):
    def __init__(
        self,
        *,
        model: str | None = None,
        prompt_version: str = "v1",
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        cache_mode: str = "record",
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
        )
        if prompt_version not in PROMPT_REGISTRY:
            raise ValueError(
                f"Unsupported prompt_version: {prompt_version}. Expected one of: {sorted(PROMPT_REGISTRY)}"
            )

        self.model = model or get_analysis_settings().llm_analysis_model
        self.prompt_version = prompt_version

    def invoke(
        self,
        *,
        question: str,
        request_options: dict[str, object] | None = None,
    ) -> DifficultyAnalysisResult:
        system_prompt = PROMPT_REGISTRY[self.prompt_version]
        user_prompt = DIFFICULTY_ANALYSIS_USER_PROMPT.format(question=question)
        runtime_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **(request_options or {}),
        )
        response_text = self.extract_chat_text(runtime_response)
        return DifficultyAnalysisResult(
            response_text=response_text,
            summary=self.extract_summary(response_text),
            input_token_count=_extract_prompt_tokens(runtime_response),
        )


def _extract_prompt_tokens(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    prompt_tokens = (
        usage.get("prompt_tokens")
        if isinstance(usage, dict)
        else getattr(usage, "prompt_tokens", None)
    )
    if isinstance(prompt_tokens, bool):
        return None
    if isinstance(prompt_tokens, int):
        return prompt_tokens if prompt_tokens >= 0 else None
    return None

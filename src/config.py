"""Centralized configuration, weights, thresholds, and skill taxonomies."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

def _load_env_file(env_path: Path | None = None) -> None:
    """Load key=value pairs from .env using python-dotenv if available, with a pure-stdlib fallback."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path)
    except Exception:
        pass

    candidates = [env_path] if env_path else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidates:
        if path and path.is_file():
            try:
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if val and val[0] in ("'", '"'):
                        quote_char = val[0]
                        end_quote = val.find(quote_char, 1)
                        if end_quote != -1:
                            val = val[1:end_quote]
                    elif " #" in val:
                        val = val.split(" #", 1)[0].strip()
                    if key and val and not os.environ.get(key):
                        os.environ[key] = val
            except Exception:
                pass


_load_env_file()


# Category maximum weights (Total = 100 points)
WEIGHT_AI_PROJECT_DEPTH: int = 40
WEIGHT_PYTHON_BACKEND: int = 30
WEIGHT_CLOUD_FULLSTACK: int = 15
WEIGHT_GITHUB_ACTIVITY: int = 10
WEIGHT_ENGINEERING_DEPTH: int = 5
TOTAL_MAX_SCORE: int = (
    WEIGHT_AI_PROJECT_DEPTH
    + WEIGHT_PYTHON_BACKEND
    + WEIGHT_CLOUD_FULLSTACK
    + WEIGHT_GITHUB_ACTIVITY
    + WEIGHT_ENGINEERING_DEPTH
)

# Project-quality penalty bounds (5-15 points as specified in Section 4)
MIN_WRAPPER_PENALTY: int = 5
MAX_WRAPPER_PENALTY: int = 15


# Canonical skill patterns: (Display Name, Regex pattern, Category)
SKILL_PATTERNS: List[Tuple[str, str, str]] = [
    # Python & Backend
    ("Python", r"\bpython\b", "python_backend"),
    ("FastAPI", r"\bfast\s*api\b", "python_backend"),
    ("Django", r"\bdjango\b", "python_backend"),
    ("Flask", r"\bflask\b", "python_backend"),
    ("Asyncio", r"\b(?:asyncio|async\s+python|aiohttp|asyncpg)\b", "python_backend"),
    ("PostgreSQL", r"\b(?:postgresql|postgres)\b", "python_backend"),
    ("Redis", r"\bredis\b", "python_backend"),
    ("SQL", r"\b(?:sql|mysql|sqlite)\b", "python_backend"),
    ("REST APIs", r"\brest(?:ful)?\s*api", "python_backend"),
    ("GraphQL", r"\bgraphql\b", "python_backend"),
    ("Pydantic", r"\bpydantic\b", "python_backend"),
    ("SQLAlchemy", r"\bsqlalchemy\b", "python_backend"),
    ("Celery", r"\bcelery\b", "python_backend"),
    # AI / Agentic / RAG
    ("LangChain", r"\blangchain\b", "ai"),
    ("LangGraph", r"\blangg?raph\b", "ai"),
    ("LlamaIndex", r"\bllama\s*index\b", "ai"),
    ("Google ADK", r"\b(?:google\s+adk|agent\s+development\s+kit)\b", "ai"),
    ("CrewAI", r"\bcrew\s*ai\b", "ai"),
    ("AutoGen", r"\bautogen\b", "ai"),
    ("LiveKit Agents", r"\blivekit\b", "ai"),
    ("MCP", r"\b(?:model\s+context\s+protocol|mcp)\b", "ai"),
    ("RAG", r"\b(?:rag|retrieval[\s-]+augmented[\s-]+generation|graph\s*rag)\b", "ai"),
    ("AI Agents", r"\b(?:ai\s+agents?|agentic|multi[\s-]+agent|tool[\s-]+calling\s+agent|autonomous\s+agents?)\b", "ai"),
    ("pgvector", r"\bpgvector\b", "ai"),
    ("ChromaDB", r"\bchroma(?:db)?\b", "ai"),
    ("Pinecone", r"\bpinecone\b", "ai"),
    ("Qdrant", r"\bqdrant\b", "ai"),
    ("FAISS", r"\bfaiss\b", "ai"),
    ("Weaviate", r"\bweaviate\b", "ai"),
    ("Milvus", r"\bmilvus\b", "ai"),
    ("Vector Search", r"\b(?:vector\s+database|vector\s+search|vector\s+store|semantic\s+search|semantic\s+retrieval)\b", "ai"),
    ("Embeddings", r"\bembeddings?\b", "ai"),
    ("OpenAI API", r"\b(?:openai|gpt-4[o]?|gpt-3\.5)\b", "ai"),
    ("Gemini", r"\bgemini\b", "ai"),
    ("Groq", r"\bgroq\b", "ai"),
    ("Ollama", r"\bollama\b", "ai"),
    ("vLLM", r"\bvllm\b", "ai"),
    ("Hugging Face", r"\bhugging\s*face\b", "ai"),
    ("Transformers", r"\btransformers?\b", "ai"),
    ("PyTorch", r"\bpytorch\b", "ai"),
    ("TensorFlow", r"\btensorflow\b", "ai"),
    ("Scikit-learn", r"\b(?:scikit[\s-]+learn|sklearn)\b", "ai"),
    # Cloud / Deployment / Full Stack
    ("GCP", r"\b(?:gcp|google\s+cloud|vertex\s+ai|cloud\s+run)\b", "cloud"),
    ("AWS", r"\b(?:aws|amazon\s+web\s+services|ec2|s3|aws\s+lambda|ecs|fargate|bedrock)\b", "cloud"),
    ("Azure", r"\b(?:azure)\b", "cloud"),
    ("Docker", r"\bdocker\b", "cloud"),
    ("Kubernetes", r"\b(?:kubernetes|k8s)\b", "cloud"),
    ("CI/CD", r"\b(?:ci/cd|github\s+actions|jenkins|gitlab\s+ci|argocd)\b", "cloud"),
    ("React", r"\breact(?:\.js|js)?\b", "fullstack"),
    ("Next.js", r"\bnext(?:\.js|js)\b", "fullstack"),
    ("TypeScript", r"\btypescript\b", "fullstack"),
    ("Node.js", r"\bnode(?:\.js|js)?\b", "fullstack"),
    # Engineering Depth
    ("Pytest", r"\bpytest\b", "engineering"),
    ("Automated Testing", r"\b(?:unit\s+tests?|unit\s+testing|integration\s+tests?|integration\s+testing|playwright|selenium|jest|junit)\b", "engineering"),
    ("Kafka", r"\bkafka\b", "engineering"),
    ("RabbitMQ", r"\brabbitmq\b", "engineering"),
    ("BullMQ", r"\bbullmq\b", "engineering"),
    ("Observability", r"\b(?:grafana|prometheus|loki|opentelemetry|langsmith|kibana|cloudwatch)\b", "engineering"),
    ("Microservices", r"\bmicroservices?\b", "engineering"),
    # Other languages (useful for rejected candidate matched_skills reporting)
    ("Java", r"\bjava\b", "other"),
    ("Spring Boot", r"\bspring\s*boot\b", "other"),
    ("Go", r"\b(?:golang)\b", "other"),
    ("C++", r"\bc\+\+", "other"),
    ("JavaScript", r"\bjavascript\b", "other"),
    ("MongoDB", r"\bmongodb\b", "other"),
]


@dataclass
class ScreeningConfig:
    """Runtime configuration for the screening pipeline."""

    input_dir: Path = field(default_factory=lambda: Path("./resumes"))
    output_path: Path = field(default_factory=lambda: Path("./output/results.json"))
    cache_dir: Path = field(default_factory=lambda: Path("./.cache"))

    # Concurrency & Networking
    max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "5"))
    )
    github_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("GITHUB_TIMEOUT_SEC", "8.0"))
    )
    llm_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SEC", "20.0"))
    )

    # Feature flags
    enable_github: bool = field(
        default_factory=lambda: os.getenv("ENABLE_GITHUB", "true").lower() in ("1", "true", "yes")
    )
    enable_llm: bool = field(
        default_factory=lambda: os.getenv("ENABLE_LLM", "true").lower() in ("1", "true", "yes")
    )
    use_cache: bool = field(
        default_factory=lambda: os.getenv("USE_CACHE", "true").lower() in ("1", "true", "yes")
    )

    # API Keys & Provider settings (always read from environment variables, never hardcoded)
    github_token: str | None = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN") or None
    )
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower()
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    )
    groq_api_key: str | None = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or None
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    )

    # Category weights
    weights: Dict[str, int] = field(
        default_factory=lambda: {
            "ai_project_depth": WEIGHT_AI_PROJECT_DEPTH,
            "python_backend": WEIGHT_PYTHON_BACKEND,
            "cloud_fullstack": WEIGHT_CLOUD_FULLSTACK,
            "github": WEIGHT_GITHUB_ACTIVITY,
            "engineering_depth": WEIGHT_ENGINEERING_DEPTH,
        }
    )

    @property
    def github_cache_path(self) -> Path:
        return self.cache_dir / "github_cache.json"

    @property
    def llm_cache_path(self) -> Path:
        return self.cache_dir / "llm_cache.json"

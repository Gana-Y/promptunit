FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/Gana-Y/promptunit"
LABEL org.opencontainers.image.description="PromptUnit: Unit testing framework for LLM prompts"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENTRYPOINT ["promptunit"]
CMD ["--help"]

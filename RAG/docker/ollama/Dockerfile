# Ollama with the ONE model the app uses baked in (config.GEN_MODEL and
# config.GATE_MODEL are both qwen3:1.7b). The RAGAS judge is NOT included.
ARG OLLAMA_TAG
FROM ollama/ollama:${OLLAMA_TAG}
RUN ollama serve >/dev/null 2>&1 & \
    until ollama list >/dev/null 2>&1; do sleep 1; done && \
    ollama pull qwen3:1.7b

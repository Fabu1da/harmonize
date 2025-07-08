# metrics.py
from prometheus_client import Counter, Histogram, start_http_server

# start the metrics HTTP server on port 8000
start_http_server(8000)

# count total OpenAI API calls, tagged by type
OPENAI_CALLS = Counter(
    "openai_calls_total",
    "Total number of OpenAI API calls",
    ["api_type"]  # e.g. embeddings vs chat
)

# latency histogram of API calls
OPENAI_LATENCY = Histogram(
    "openai_call_duration_seconds",
    "Latency of OpenAI API calls",
    ["api_type"]
)

# count of pipeline failures
PIPELINE_ERRORS = Counter(
    "pipeline_errors_total",
    "Number of uncaught exceptions in pipeline"
)

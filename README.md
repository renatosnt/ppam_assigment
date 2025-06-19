configs/ centralizes experiment settings—everyone runs with the same defaults and can override via CLI.

data/raw vs. data/processed separates immutable source files from client-partitioned parquet for fast loading.

src/ modularization keeps preprocessing, modeling, FL glue code, and evaluation in separate files for easy testing and review.

notebooks/ for iterative exploration and demo runs—no production code here.

tests/ ensure each module works in isolation (e.g. data splits, parameter packing).

docker-compose.yml + Dockerfile let anyone on the team spin up the server + N clients with one command:
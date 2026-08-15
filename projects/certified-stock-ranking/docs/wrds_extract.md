# WRDS extraction

The preferred production extract is the prebuilt JKP table described in
`docs/jkp_data.md`. The legacy OSAP-plus-CRSP pipeline remains in the repository
only as an independent replication robustness check.

Credentials must be supplied through the process environment. Never place a
WRDS username or password in Python files, notebooks, YAML, shell scripts,
GitHub Actions, `.env` files, or committed `.pgpass` files.

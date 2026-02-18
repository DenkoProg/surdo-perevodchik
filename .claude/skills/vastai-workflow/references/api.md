# Vast.ai API Reference (Concise)

Always validate endpoint details against official docs:
https://docs.vast.ai/api-reference/introduction

This file is a working snapshot for day-to-day execution.
Refresh it when endpoint contracts, required params, or response shapes change.

## Base URL and Auth

- Use `https://console.vast.ai/api/v0/`.
- Confirm per-endpoint auth style in docs (`api_key` query param vs header).
- Keep `VAST_API_KEY` in environment; never print secret values.

## Core Endpoint Groups

1. Offers/search (`/bundles`, and any documented alternatives)
2. Ordering/create (`/asks/{id}` style flows)
3. Instance status/list (`/instances`, `/instances/{id}`)
4. SSH key management (`/ssh`, `/instances/{id}/ssh`)
5. Lifecycle controls (start/stop/restart/destroy)
6. Billing/usage/invoices

## Safe Request Checklist

- Validate request shape before execution (method, path, required params).
- For offer selection, enforce `rentable=true` and `rented=false`.
- Create one instance at a time; avoid loops that can start multiple unintended instances.
- After create, poll state and verify readiness before SSH.
- On teardown, verify resulting state is destroyed/stopped as requested.

## Common Failure Modes

- `401/403`: key invalid, missing, or wrong permissions.
- `429`: rate limited; retry with backoff.
- `404`/`4xx`: endpoint mismatch or invalid args.
- `5xx`: transient server issues; retry and then re-check resource state.
- JSON parse failures from shell pipes: save output to temp file and parse from file.

# Security Policy

## Reporting a Vulnerability

Report suspected vulnerabilities privately to the repository maintainers through an existing private channel. If no private channel has been established, contact the maintainers only to arrange one; do not disclose vulnerability details in a public issue.

Include the affected route or component, reproducible steps, realistic impact, affected deployment assumptions, and a safe redacted proof. Do not include secrets or private data in a report, including credentials, access tokens, push subscription endpoints or keys, database contents, private provider URLs, request bodies, or exploit payloads containing sensitive material.

## System and Scope

This policy covers RecLive's React/Vite PWA, FastAPI service, MySQL persistence and migrations, scheduled occupancy, forecast, and facility-hours jobs, Web Push rule management and dispatch, compatibility entry points under `server/`, and repository-owned CI and deployment configuration.

RecLive's supported deployment model treats the public dashboard and public API routes as internet-facing application boundaries; this is a security assumption for review, not a claim that a particular current production deployment was verified. Admin routes are privileged boundaries when enabled. Important assets include backend configuration and credentials, database contents, browser push subscription material, integrity and freshness of occupancy, forecast, and schedule data, migration state, and the availability of supported `/nick` and `/bakke` behavior.

## Threat Model and Trust Boundaries

- Treat browser requests, query and path values, push subscriptions, cached browser data, upstream occupancy responses, facility-hours HTML or JSON, forecast and schedule artifacts, HTTP headers, and provider responses as untrusted input.
- The frontend bundle is public. Only explicitly public build settings may enter it; database settings, upstream URLs, admin tokens, endpoint-hash keys, VAPID private material, and other backend settings must stay on the backend.
- The deployment secret manager or backend process environment and authorized operators are trusted to supply private settings. Their presence does not make a value safe to log, return, commit, or copy into a frontend artifact.
- MySQL, the filesystem, schedulers, DNS, and external providers are dependencies that can be unavailable, stale, malformed, or misconfigured. Their output must be validated at the consuming boundary.
- Public push-rule ownership is established from the validated subscription rather than browser local storage. Admin mutations require the configured admin authorization boundary.

## Security Invariants

The following properties are requirements for supported deployments; this policy does not claim that documentation or tests alone prove them:

- Production configuration must fail closed when required private settings are absent or placeholder-like, CORS is wildcarded, or enabled admin routes lack a strong token. Validation errors may identify variable names and reasons but must not reveal rejected values.
- Credentials, production database dumps or records, VAPID private keys, endpoint-hash keys, admin tokens, credential-bearing provider URLs, real browser push endpoints or keys, generated secrets, private request or provider data, and private history-rewrite bundles must not enter source, frontend bundles, logs, test fixtures, documentation, issues, or committed artifacts. Clearly synthetic nonproduction test fixtures and explicitly non-working placeholders are permitted when they contain no production or private material and cannot authenticate to production or third-party services.
- Public and admin requests must be bounded and schema-validated before use. Authorization and ownership checks must occur before protected reads or mutations, and comparisons of secret authorization material must resist timing leaks.
- Outbound provider and push targets must be canonicalized and constrained to their intended HTTPS and network boundaries. User-controlled URLs must not provide access to loopback, private, link-local, or otherwise disallowed destinations.
- Push-rule creation, ownership, cancellation, rate limiting, claiming, and terminal transitions must preserve their transactional and cross-worker safety properties. API responses and errors must not expose subscription endpoints or keys.
- Invalid, empty, future-dated, unavailable, or stale upstream data and artifacts must not be represented as validated fresh data. Fresh observed rows may coexist with partial coverage, but missing or stale portions must not be invented as zero or represented as complete or fully fresh coverage; the partial scope must remain explicit. Failed refreshes must not overwrite validated last-known-good facility data, and health state must identify the failing boundary without inventing evidence.
- Health endpoints must remain read-only and expose only bounded status, timestamps, ages, counts, and fixed safe categories. They must not expose hosts, database names, credentials, upstream URLs, keys, tokens, endpoints, bodies, environment values, exception text, or traces.
- Operational logs must use defined event names and allowlisted, bounded scalar fields. Environment dictionaries, SQL values, raw provider responses, raw push subscriptions, request bodies, generated secrets, raw exceptions, and other untrusted values must not cross the logging boundary.
- Applied migrations are immutable and checked by the migration runner using the effective migration filenames and checksums. Health probes must be read-only and must not repair or alter migration state.
- The service worker and browser cache must not treat API responses as application assets. Notification navigation must remain on the configured application origin.

## Reportable Findings and Severity Context

A finding is reportable when it demonstrates a realistic path to violate an invariant or creates meaningful confidentiality, integrity, authorization, privacy, or availability impact in a supported deployment. Assess severity from actual reachability, required privileges, exposed data, persistence, cross-user or cross-worker impact, and the reliability of the demonstrated path.

Examples include authentication or ownership bypass, secret or push-subscription disclosure, server-side request forgery, injection, unsafe deserialization or parsing, persistent frontend injection or cache poisoning, migration-integrity bypass, rate-limit or request-bound bypass with security impact, unauthorized push dispatch or rule mutation, and data-trust failures that cause untrusted or stale information to be published as validated fresh state.

Scanner output, dependency advisories, and tests are evidence to investigate. They do not by themselves prove exploitability or safety, and a missing test is not proof that a control is absent.

## Out of Scope, Exclusions, and Accepted Risk

No vulnerability class, repository-owned component, or supported deployment is intentionally excluded by this policy, and this policy declares no accepted security risks. Any future exclusion, severity exception, or accepted risk requires explicit maintainer approval and documented rationale; a known limitation or compensating control is not suppression authority.

## Secret Handling and Credential Rotation

Use ignored `.env` files for local private settings and the deployment platform's secret manager or process manager for deployed private settings. `.env.example` may contain names and non-working placeholders only. Never commit credentials, dumps, private backup bundles, or artifacts containing private data.

If a credential may have been exposed, credential rotation must occur through the owning provider or secret manager: contain access, rotate or revoke the credential, update the deployed secret out of band, and verify the application without printing the old or new value. Do not paste credential material into commits, issues, chat transcripts, test fixtures, logs, shell commands, or documentation. Follow the provider's audit and revocation procedure without changing unrelated credentials.

## History Rewrite Limitations

The repository underwent a history rewrite to remove historical credential values from reachable published branches and tags. A history rewrite does not erase copies retained in forks, caches, old clones, backup bundles, third-party indexes, local reflogs, or other storage outside the rewritten references.

Collaborators must re-clone or carefully rebase clean work onto the rewritten history and must not republish old references. Any private pre-rewrite backup must remain untracked, access-controlled, and outside ordinary repository tooling. Rewrite and scan results are evidence about the references checked, not proof that every external copy was erased.

## Security Verification

The repository security workflow runs redacted Gitleaks scanning over all reachable history on pull requests and on pushes to `main` and `hardening/reclive-security-data-trust`. Dependency review runs on pull requests only. These checks complement, but do not replace, review of trust boundaries and runtime behavior.

Before a release, run the documented frontend, backend, migration, PWA, and browser checks; run Gitleaks against all reachable refs with redacted output; run `npm audit --omit=dev`; and run `git diff --check`. Record commands actually executed and their results. Mark provider-backed or credential-backed checks as unexecuted when they were unavailable; do not infer success from mocks or workflow configuration. Investigate findings without copying suspected secret material into tickets or logs.

## Known Limitations and Compensating Controls

- A sanitized health response reports local readiness and freshness evidence. It does not prove external provider availability, successful Web Push delivery, or end-to-end user receipt.
- Gitleaks and dependency automation reduce risk but cannot prove the absence of secrets or reachable vulnerable code. Findings and advisories require evidence-based triage, and unresolved items remain visible until addressed or explicitly accepted by maintainers.
- A successful history rewrite and fresh-clone check apply to the references and copies examined; the external-copy limitations above continue to apply.
- Validation, rate limiting, transactional state changes, sanitized errors and logs, strict frontend schemas, and fail-closed production configuration are compensating controls to verify, not assumptions that automatically lower severity.

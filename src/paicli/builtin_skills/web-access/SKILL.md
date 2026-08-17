---
name: web-access
description: |
  Research current information on the public web or in a browser session. Use for live facts, recent changes, source-backed answers, webpage inspection, dynamic or login-gated pages, and social sites.
---

# Web Access

Follow this sequence:

1. Identify which claims require current evidence and what date range matters.
2. Use `web_search` to discover candidate sources; prefer primary and authoritative sources.
3. Use `web_fetch` for public static pages. Open the specific source rather than relying only on a search snippet.
4. Use Chrome DevTools MCP only when the task needs an existing login, JavaScript rendering, browser interaction, or visible page inspection.
5. Cross-check consequential claims when one source may be stale, ambiguous, or promotional.
6. Report source URLs and relevant dates near the claims they support. Distinguish sourced facts from inference.

If a static fetch is empty or blocked, switch to the browser path when available. If no usable path is available, state the limitation and do not guess.

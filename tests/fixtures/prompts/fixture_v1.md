<!-- Test fixture only: exercises the loader and the loop. Not a real prompt. -->
## system
Answer as JSON.

## user
Repo {{ repo }}.
{{ needs_llm_text }}
Targets: {{ candidates.target_column | join(", ") }}

## retry
Attempt {{ attempt }}. Rejected:
{{ reasons_text }}

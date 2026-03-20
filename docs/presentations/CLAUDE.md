# Presentation Workshop

This folder holds HTML slide decks. Each presentation is a single self-contained `.html` file that opens in Chrome or Claude's artifact viewer.

## How to use

Give me a rough idea and I'll produce a complete, polished HTML presentation in one shot. Then we iterate with feedback until it's exactly right.

**IMPORTANT: Always invoke the `frontend-slides` skill (via the Skill tool) when creating or significantly revising a presentation. Never generate presentation HTML without it.**

**Example prompts:**
- "Make a 6-slide pitch deck for our Ukrainian sign-language translator: problem, solution, demo, model results, roadmap, call to action"
- "Create a technical deep-dive into our fine-tuning pipeline, dark theme, 8 slides"
- "Short 4-slide intro for a university defense, formal style"

---

## Skill: `frontend-slides`

Every presentation request must go through the `frontend-slides` skill. This skill:
- Produces stunning, animation-rich HTML presentations from scratch
- Handles aesthetic exploration, motion design, and visual hierarchy automatically
- Converts ideas or outlines into production-quality single-file slide decks

**Trigger conditions (use the skill whenever):**
- User asks to create a presentation, slide deck, or pitch
- User asks to revise or restyle an existing presentation significantly
- User provides a PPT/PPTX to convert to web format

Do not skip the skill and generate presentation HTML manually.

---

## Iteration Protocol

After the first draft:
1. Open the file in Chrome (`Cmd/Ctrl+O` - select the `.html`) or paste contents into Claude's artifact viewer
2. Give feedback slide-by-slide or globally ("slide 3 needs more impact", "change palette to warm earth tones", "add a chart to slide 5")
3. For visual/style changes, re-invoke `frontend-slides`; for small copy edits, edit directly

---

## File Naming

`YYYY-MM-DD_<short-slug>.html` - e.g., `2026-03-20_model-results.html`

Save finished presentations here: `docs/presentations/`

# Decisions Log

## Purpose
Decision index for stable project choices and rationale.

## Decision Log
- Add links to individual decision notes using `[[DEC-YYYYMMDD-short-slug]]`.
- **2026-04-24 — Tech comparison subs in `primary`:** Added six non-political forums for contrast with political discourse: coding `learnprogramming`, `AskProgramming`, `CodingHelp`; career `cscareerquestions`, `ITCareerQuestions`, `csMajors`. Most extra storage vs prior political-only scope comes from `learnprogramming` and `cscareerquestions`. After changing `subreddits.primary`, delete `results/logs/filter_dump/filter_dump_state.RC_*.json` (and merged `filter_dump_state.json` if used) before re-filtering so completed dump months are rescanned.
- **2026-04-24 — General Q&A subs in `primary`:** Added `answers`, `TooAfraidToAsk`, `OutOfTheLoop` for answer-format and topic diversity (mid-size vs default megas). Same filter-state reset rule applies when expanding `subreddits.primary`.

## Links
- [[Projects/Thesis-Project-Hub]]

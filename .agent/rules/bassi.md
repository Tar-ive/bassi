---
trigger: model_decision
description: After a feature has been completed and a push to github has been made. 
---

# Major Decisions Ruleset

**TRIGGER**: Ruleset activates after feature completion and GitHub push.

## Core Requirements
- **MANDATORY**: Verify `major_decisions.json` exists in repository root
- **DATE SOURCE**: Scan `/Users/tarive/.gemini/antigravity/brain/` for current working date
- **COMMIT HASHES**: ALL "Implemented" decisions MUST include valid commit hashes
- **ID FORMAT**: Sequential 3-digit numbers (001, 002, 003...)

## Decision Schema
```json
{
  "id": "sequential_3digit_string",
  "date": "YYYY-MM-DD_format_only",
  "decision": "clear_concise_title",
  "context": "detailed_explanation_and_rationale",
  "status": "Accepted|Implemented|Planned",
  "commit": "full_git_commit_hash_required"
}
```

## Status Progression
1. **Accepted**: Initial architectural approval
2. **Implemented**: Code committed with hash
3. **Planned**: Future work identified

## Workflow
1. **Pre-work**: Confirm file exists, scan brain directory, validate commit hashes
2. **During**: Document decisions with proper ID, date, context
3. **Post-commit**: Update status to "Implemented", capture commit hash
4. **Post-push**: Validate all implemented decisions have valid hashes

## Error Handling
- **Missing file**: "Cannot proceed without major_decisions.json"
- **Missing commit hash**: "CRITICAL: Decision [ID] missing commit hash"
- **Invalid date**: "Must use YYYY-MM-DD format"
- **ID conflict**: "IDs must be sequential 3-digit numbers"

## Git Integration
- Capture commit hashes immediately after implementation commits
- Validate hashes exist in repository
- Reference decision IDs in commit messages when applicable
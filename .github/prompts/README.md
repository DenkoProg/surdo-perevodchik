## Resources

- [VS Code Prompt Files Documentation](https://code.visualstudio.com/docs/copilot/customization/prompt-files)
- [Community Examples](https://github.com/github/awesome-copilot)
- [Custom Instructions](https://code.visualstudio.com/docs/copilot/customization/custom-instructions)
- [Chat Modes](https://code.visualstudio.com/docs/copilot/customization/custom-chat-modes)

---

## Overview

Prompt files (`.prompt.md`) are reusable Markdown files that define standardized prompts for common development tasks. They enable you to create a library of workflows that can be triggered on-demand in VS Code Chat.

**Key Benefits:**

- ✅ Reusable across projects
- ✅ Standardized development workflows
- ✅ Support for variables and parameters
- ✅ Can reference other files and workspaces
- ✅ Shareable with team via git

## File Location

Store workspace prompt files in:

```
.github/prompts/
```

## Naming Convention

**Use kebab-case starting with an action verb:**

```
generate-tasks.prompt.md           # Generate task breakdown
migrate-api.prompt.md              # Migrate API patterns
review-code.prompt.md              # Review code changes
create-tests.prompt.md             # Create test suites
refactor-component.prompt.md       # Refactor components
analyze-performance.prompt.md      # Analyze performance
document-api.prompt.md             # Document API endpoints
scaffold-feature.prompt.md         # Scaffold new features
optimize-queries.prompt.md         # Optimize database queries
deploy-service.prompt.md           # Deploy service to cloud

```

**Pattern:** `{verb}-{object}.prompt.md`

**Common verbs:** generate, create, migrate, refactor, review, analyze, document, scaffold, optimize, deploy, test, fix, update, validate, build

## Template for New Prompts

```markdown
---
description: [What this prompt does]
mode: agent # or 'ask' or 'edit'
---

# [Optional: Title]

${input:mainInput:Description of what user should provide}

You are a [role/expertise].

Your task:
[Clear instructions]

Requirements:

- [Requirement 1]
- [Requirement 2]

Output:
[Expected format/deliverable]

Reference files: [file.ts](../path/to/file.ts)
Reference other prompts: [other.prompt.md](./other.prompt.md)
```

## Usage in VS Code Chat

### Method 1: Slash Command (Recommended)

```
/prompt-name
```

VS Code will prompt you for any input variables.

### Method 2: Inline Parameters

```
/prompt-name: var1="value1", var2="value2"
```

### Method 3: Command Palette

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Run `Chat: Run Prompt`
3. Select your prompt from the list

### Method 4: Editor Play Button

1. Open the `.prompt.md` file in VS Code
2. Click the ▶️ play button in the editor title
3. Choose to run in current or new chat session

---

## Example: Task Generation Prompt

### File: `.github/prompts/generate-tasks.prompt.md`

```yaml
---
description: Generate a tasks.md file for migrations or implementations
mode: agent
---

${input:targetRepo:@workspace} (target) and ${input:exampleSource:@workspace or /examples or "none"} (example) are the repositories/sources to focus on.

You are a Senior ML Architect and MLOps engineer.

Solve this task:
${input:task:Describe the task (e.g., "migrate from BentoML to Triton")}

Do it in the cleanest and best practice way. Write a detailed tasks.md file and save it into the target repo.

Develop a clear, step-by-step plan. Break down the changes into manageable, incremental steps. Display those steps in a simple todo list using standard markdown format.

```

### How to Use

**In VS Code Chat, type:**

```
/tasks-generation
```

**VS Code will prompt for:**

1. **targetRepo**:
    - Enter: `@bentoml-rafflesia-main-classificator`
    - Or: `@workspace` (for current repo)
2. **exampleSource**:
    - Enter: `@bentoml-rock-identifier` (reference workspace)
    - Or: `/examples` (local examples folder)
    - Or: `"none"` (no reference needed)
3. **task**:
    - Enter: `migrate from BentoML to Triton using ONNX and GPU acceleration`

**Alternative - All at Once:**

```
/tasks-generation: targetRepo=@workspace, exampleSource=@bentoml-rock-identifier, task="migrate from BentoML to Triton"
```

## Best Practices

### ✅ DO:

- Use clear, descriptive names
- Include helpful placeholder text in variables
- Keep prompts focused on one task
- Test prompts before committing
- Document complex prompts with examples
- Use `mode: agent` for file-creating tasks
- Reference workspace files with relative paths

### ❌ DON'T:

- Make prompts too generic or vague
- Hard-code values that should be variables
- Create extremely long prompts (split into multiple files)
- Forget to add description metadata
- Use absolute paths

---

## Advanced Patterns

### Chain Multiple Prompts

Reference other prompts:

```markdown
First, run [setup.prompt.md](./setup.prompt.md) to initialize the project.

Then proceed with the implementation:
[Your instructions here]
```

### Reference Instructions Files

Link to custom instructions:

```markdown
Follow the guidelines in [instructions.md](../copilot-instructions.md).

[Your prompt content]
```

---

## Enabling Prompt Files (If not enabled by default)

1. Open VS Code Settings (`Cmd+,` or `Ctrl+,`)
2. Enable: `chat.promptFiles`
3. Configure additional locations: `chat.promptFilesLocations`

---
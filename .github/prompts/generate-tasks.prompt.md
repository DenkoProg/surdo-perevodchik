---
description: Generate a tasks.md file for migrations or implementations
mode: agent
---

${input:targetRepo:@workspace} (target) and ${input:exampleSource:@workspace or /examples or "none"} (example) are the repositories/sources to focus on.

You are a Senior ML Architect and MLOps engineer.

Solve this task:
${input:task:Describe the task (e.g., "Move the project from BentoML to Triton")}

Do it in the cleanest and best practice way. Write a detailed tasks.md file and save it into the target repo.

Develop a clear, step-by-step plan. Break down the changes into manageable, incremental steps. Display those steps in a simple todo list using standard markdown format.
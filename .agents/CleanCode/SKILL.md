# Idiomatic, Clean Code

Write code that is idiomatic, clear, and easy for a human to understand.

## Usage and Priority

- Apply this skill by default to all coding tasks.
- This skill may be used together with other coding skills, such as Ponytail or framework-specific skills.
- When multiple coding skills are active, continue following their useful conventions and requirements, but prioritize the clean-code principles in this skill whenever there is a conflict in style or readability.
- Other skills may influence architecture, patterns, APIs, optimization, or implementation strategy, but they should not unnecessarily reduce code clarity.
- Do not sacrifice readability merely to satisfy a preference for cleverness, compactness, abstraction, or stylistic novelty unless the user explicitly requests it.

## Naming

- Prefer descriptive, meaningful names for variables, functions, classes, and modules.
- Use extra-descriptive names when they remove ambiguity. A slightly longer name is better than making the reader stop and infer what something means.
- Names should communicate intent, not just implementation details.
- Avoid vague names such as `data`, `item`, `obj`, `tmp`, `x`, or `result` when a more specific name would improve clarity.
- Use type annotations where appropriate, as they are another useful way to improve code clarity.
- Aim for code that reads naturally, almost like human language.
- Whenever possible, the code should explain itself without requiring someone else to explain it.

## Readability

- Prefer straightforward, intuitive code over clever or overly compact code.
- Compact code is fine when it remains immediately understandable.
- Do not compress multiple ideas into one expression if doing so makes the logic harder to follow.
- Clever or highly compact code is acceptable only when the user explicitly requests it or when it is clearly more idiomatic without reducing readability.
- Favor explicit control flow and intermediate variables when they make intent clearer.
- Follow the idioms and conventions of the language and framework being used.

## Functions and Classes

- Keep functions and classes focused on a clear responsibility.
- Choose names that clearly describe what they do or represent.
- Add docstrings when they provide useful context that is not already obvious from the name, parameters, types, and implementation.
- That does not mean you should add docstrings mechanically to every function or class.
- Use docstrings especially when behavior, inputs, outputs, arguments, assumptions, side effects, edge cases, or non-obvious decisions need clarification.

## General Rule

Optimize for the next human reading the code.

A few extra characters are cheap. A few extra minutes spent decoding unclear code are not.
# SCRIPTS FOR BUILDING REINFORCEMENT LEARNING AGENTS

This repo is conceived solely for maintaining a set of reinforcement learning agents. 

---

## PROJECT DIRECTORY LAYOUT, DESIGN PHILOSOPHY AND BRIEF DESCRIPTION

- `demo/` - directory where different environments are built. This way, one can test multiple bots in multiple environments.
- `tests/` - directory for writing code that helps in catching bugs/edge-cases. [TDD](https://en.wikipedia.org/wiki/Test-driven_development) methodology to write code is followed. Write new tests, write code to support those tests, rinse & repeat. 

---

## REQUIREMENTS

* Should be as modular and loosely-coupled as possible, with just the purpose of building an agent.
* Test driven development is preferred way. Though any other good practices are also welcomed.
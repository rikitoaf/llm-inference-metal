# Notice

This repository holds configuration, scripts, tests, tools, documentation and patches for running local
Qwen3.8-27B models with a coding-agent harness. The patches apply to other people's projects, which keep
their own licenses:

| Patch | Applies to | License of that project |
|---|---|---|
| `patches/0001`, `patches/0002` | the llama.cpp fork `atomic-llama-cpp-turboquant` (a llama.cpp derivative) | MIT, in that repository's `LICENSE` |
| `patches/0003`, `splash/staged/` | [incoai/splash](https://github.com/incoai/splash) 1.2.1 | Apache License 2.0, in that repository's `LICENSE` |

The patch files contain lines of those projects as context. Those lines and the changes made to those files stay
under the license of the project they belong to. The `splash/staged/` files are copies of Splash source files.

The Swift 1.5 Qwen3.8-27B Uncensored model that README section 33 describes is not part of this repository.
It is downloaded from Hugging Face under its own license (Swift Open License v1.0; Qwen3.8 is Apache 2.0).

"""Generation entry point. Routes to MLX or PyTorch implementation.

Re-exports the public API (load, generate_stream, build_chat_prompt, CHAT_STOPS)
from the selected backend so callers can `from tasochka.generate import ...`.
"""
from ._backend import resolve

_backend = resolve()
if _backend == "mlx":
    from .generate_mlx import (  # noqa: F401
        CHAT_STOPS,
        build_chat_prompt,
        generate,
        generate_stream,
        load,
        main as _main,
    )
else:
    from .generate_torch import (  # noqa: F401
        CHAT_STOPS,
        build_chat_prompt,
        generate,
        generate_stream,
        load,
        main as _main,
    )


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

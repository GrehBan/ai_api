The provider types are reproducible: `scripts/generate_provider_types.py`
runs aiollm-codegen from the `codegen` dependency group, pinned to a
commit of its public repository, and downloads OpenAI's OpenAPI file at
a pinned commit, refusing a file whose SHA-256 is not the one of the
contract snapshot. CI regenerates the types and fails when any file
changes.

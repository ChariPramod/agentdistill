# Chat template fixtures

Three templates, used by `test_build_masks.py` to assert token-level loss masks:

| file | purpose |
|---|---|
| `toolchat.jinja` | Renders tools and tool calls, prefix-stable. The happy path. |
| `no_tools.jinja` | Ignores the `tools` argument entirely. `base-check` must reject it. |
| `unstable.jinja` | Re-renders earlier turns once a later one arrives (it appends a running turn count to every assistant header). `build_trajectory_sample` must raise `MaskingError`. |

They are deliberately spelled in ordinary text rather than added special tokens, so the tests also cover the
straddling-token case that `_labels_from_spans` handles.

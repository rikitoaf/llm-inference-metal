import re


def _strip(text):
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text


def toc(markdown, max_level=3):
    heads, fence = [], None
    for line in markdown.split("\n"):
        if fence:
            if line.startswith(fence):
                fence = None
            continue
        if line.startswith("```") or line.startswith("~~~"):
            fence = line[:3]
            continue
        m = re.match(r"^(#{1,6}) (.*)$", line)
        if not m:
            continue
        raw = re.sub(r"\s+#+\s*$", "", m[2]).strip()
        heads.append((len(m[1]), _strip(raw)))
    used, out = {}, []
    for level, text in heads:
        base = "".join(ch for ch in text.lower() if ch.isalnum() or ch in " -_").replace(" ", "-")
        n = used.get(base, 0)
        slug = base if n == 0 else f"{base}-{n}"
        used[base] = n + 1
        if level <= max_level:
            out.append((level, text, slug))
    if not out:
        return ""
    lo = min(l for l, _, _ in out)
    return "\n".join(f"{'  ' * (l - lo)}- [{t}](#{s})" for l, t, s in out)

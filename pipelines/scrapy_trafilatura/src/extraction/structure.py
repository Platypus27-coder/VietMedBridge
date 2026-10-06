"""Preserve nested headings, visible link text, lists and table group labels."""
from __future__ import annotations

import re

BLOCKS = {'p', 'div', 'section', 'article', 'main', 'body', 'blockquote', 'figure', 'figcaption', 'caption', 'pre', 'ab'}
SKIP = {'script', 'style', 'noscript', 'iframe', 'form', 'button', 'input', 'svg', 'nav', 'footer'}


def tag(node):
    return node.tag.lower().split('}')[-1] if isinstance(node.tag, str) else ''


def tidy(text):
    return re.sub(r'[ \t\r\f\v]+', ' ', text).strip()


def heading_level(node):
    name = tag(node)
    if re.fullmatch(r'h[1-6]', name):
        return int(name[1])
    if name == 'head':
        match = re.search(r'h([1-6])', node.get('rend', ''))
        return int(match.group(1)) if match else 2
    return None


def inline(node, markdown=False):
    if not tag(node) or tag(node) in SKIP:
        return ''
    parts = [node.text or '']
    for child in node:
        parts.append('\n' if tag(child) == 'br' else inline(child, markdown))
        parts.append(child.tail or '')
    text, name = ''.join(parts), tag(node)
    if markdown and text.strip():
        if name in ('b', 'strong') or name == 'hi' and 'bold' in node.get('rend', ''):
            text = '**' + text.strip() + '**'
        elif name in ('em', 'i') or name == 'hi' and 'italic' in node.get('rend', ''):
            text = '*' + text.strip() + '*'
    return text


def table(node, markdown):
    rows = [r for r in node.iter() if tag(r) in ('tr', 'row')
            and next((a for a in r.iterancestors() if tag(a) == 'table'), None) is node]
    if len(rows) > 2000:
        raise OverflowError('Table exceeds 2,000 rows.')
    grid, spans = [], {}
    for index, row in enumerate(rows):
        values = {col: value for col, (until, value) in spans.items() if until > index}
        column = 0
        for cell in row:
            if tag(cell) not in ('td', 'th', 'cell'):
                continue
            while column in values:
                column += 1
            value = tidy(inline(cell, markdown)).replace('\n', ' / ')
            def span(name):
                try:
                    return max(1, min(100, int(cell.get(name, '1'))))
                except ValueError:
                    return 1
            for offset in range(span('colspan')):
                if column + offset >= 100:
                    raise OverflowError('Table exceeds 100 columns.')
                values[column + offset] = value
                if span('rowspan') > 1:
                    spans[column + offset] = (index + span('rowspan'), value)
            column += span('colspan')
        grid.append(values)
    width = max((max(row, default=-1)+1 for row in grid), default=0)
    caption = '\n\n'.join(tidy(inline(c, markdown)) for c in node if tag(c) == 'caption')
    if not width:
        return caption or tidy(inline(node, markdown))
    lines = []
    for i, values in enumerate(grid):
        cells = [values.get(col, '') for col in range(width)]
        if markdown:
            lines.append('| ' + ' | '.join(c.replace('|', '\\|') for c in cells) + ' |')
            if i == 0:
                lines.append('| ' + ' | '.join('---' for _ in range(width)) + ' |')
        else:
            lines.append(' | '.join(cells))
    return '\n\n'.join(part for part in (caption, '\n'.join(lines)) if part)


def render(node, markdown=False, depth=0):
    name = tag(node)
    if not name or name in SKIP:
        return ''
    level = heading_level(node)
    if level:
        return ('#' * level + ' ' if markdown else '') + tidy(inline(node, False))
    if name == 'table':
        return table(node, markdown)
    if name in ('ul', 'ol', 'list'):
        items = []
        for index, child in enumerate(c for c in node if tag(c) in ('li', 'item')):
            lines = render_children(child, markdown, depth+1).splitlines()
            marker = f'{index+1}. ' if name == 'ol' else '- '
            if lines:
                items.append(('  '*depth + marker if markdown else '') + lines[0] +
                             ''.join('\n' + ('  ' if markdown else '') + line for line in lines[1:]))
        return '\n'.join(items)
    if name == 'br':
        return '\n'
    return render_children(node, markdown, depth)


def render_children(node, markdown=False, depth=0):
    chunks, pending = [], [node.text or '']
    def flush():
        value = tidy(''.join(pending))
        if value:
            chunks.append(value)
        pending.clear()
    for child in node:
        name = tag(child)
        if heading_level(child) or name in BLOCKS | {'table', 'ul', 'ol', 'list', 'li', 'item'}:
            flush()
            value = render(child, markdown, depth)
            if value.strip():
                chunks.append(value)
        elif name == 'br':
            pending.append('\n')
        else:
            pending.append(inline(child, markdown))
        pending.append(child.tail or '')
    flush()
    return '\n\n'.join(chunks)


def serialize(nodes):
    return {'text': '\n\n'.join(render(node) for node in nodes).strip(),
            'text_markdown': '\n\n'.join(render(node, True) for node in nodes).strip()}

import sys

MOD = 'crates/onnx-genai-engine/src/pipeline/mod.rs'

def read():
    with open(MOD) as f:
        return f.read().split('\n')

def write(lines):
    with open(MOD, 'w') as f:
        f.write('\n'.join(lines))

def extract(ranges, out_path, header, wrap_impl=True):
    """ranges: list of (start,end) 1-based inclusive. Removes from MOD, writes out_path."""
    lines = read()
    # collect
    chunks = []
    for (s, e) in ranges:
        chunks.append('\n'.join(lines[s-1:e]))
    body = '\n\n'.join(chunks)
    content = header
    if wrap_impl:
        content += '\nimpl PipelineEngine {\n' + body + '\n}\n'
    else:
        content += '\n' + body + '\n'
    with open(out_path, 'w') as f:
        f.write(content)
    # remove ranges from MOD, bottom-up
    for (s, e) in sorted(ranges, reverse=True):
        del lines[s-1:e]
    write(lines)

if __name__ == '__main__':
    pass

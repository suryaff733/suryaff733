import sys
from pathlib import Path
from html import escape
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

def generate_ascii(image_path, cols=110, rows=53):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=float)

    # Detect subject silhouette (pure black background is < 8)
    mask = np.any(arr > 8, axis=2)

    # Find tight subject bounding box in original image
    y_idx, x_idx = np.where(mask)
    min_y, max_y = y_idx.min(), y_idx.max()

    # Frame crop: Include full hair at top down to mid-chest/shirt
    crop_top = max(0, min_y - 12)
    crop_bottom = min(img.height, crop_top + 920)
    crop_left = 0
    crop_right = img.width

    cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    cropped_mask = Image.fromarray(mask[crop_top:crop_bottom, crop_left:crop_right].astype(np.uint8) * 255)

    gray = cropped_img.convert('L')

    # 1. Luminance equalization blended with natural lighting for shadow/hair & skin tones
    eq = ImageOps.equalize(gray)
    blended = Image.blend(gray, eq, alpha=0.35)

    # 2. Dual-frequency sharpening for micro-details (glasses frames, iris, lips) + macro structure
    sharp1 = blended.filter(ImageFilter.UnsharpMask(radius=1.5, percent=250, threshold=1))
    sharp2 = sharp1.filter(ImageFilter.UnsharpMask(radius=4.0, percent=150, threshold=2))

    # 3. High-pass edge reinforcement for facial boundaries, eye contours & hair texture
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges_enhanced = ImageEnhance.Contrast(edges).enhance(1.8)
    edge_arr = np.array(edges_enhanced, dtype=float)
    base_arr = np.array(sharp2, dtype=float)
    combined = np.clip(base_arr + edge_arr * 0.25, 0, 255).astype(np.uint8)
    combined_img = Image.fromarray(combined)

    # Resize to high-resolution character grid
    resized_gray = np.array(combined_img.resize((cols, rows), Image.Resampling.LANCZOS), dtype=float)
    resized_mask = np.array(cropped_mask.resize((cols, rows), Image.Resampling.BILINEAR), dtype=float) / 255.0

    # Smooth density ramp with balanced optical density
    chars = '  ..::--==++**#%%@@'

    lines = []
    for y in range(rows):
        row_chars = []
        for x in range(cols):
            m = resized_mask[y, x]
            if m < 0.15:
                row_chars.append(' ')
            else:
                val = resized_gray[y, x]
                norm = val / 255.0
                idx = int(1 + norm * (len(chars) - 2))
                idx = max(1, min(len(chars) - 1, idx))
                row_chars.append(chars[idx])
        lines.append(''.join(row_chars))

    return lines

def build_tspans(lines, start_x=22, start_y=64.0, step_y=8.0):
    tspans = []
    y = start_y
    for line in lines:
        escaped_line = escape(line)
        tspans.append(f'<tspan x="{start_x}" y="{y:.2f}" xml:space="preserve">{escaped_line}</tspan>')
        y += step_y
    return tspans

def update_svg(svg_path, tspans, start_x=22):
    content = Path(svg_path).read_text(encoding='utf-8')
    import re
    m = re.search(r'<text x="\d+" y="0" class="ascii">', content)
    if not m:
        raise ValueError(f"Could not find ascii text tag in {svg_path}")
    start_idx = m.start()

    end_tag = '</text>'
    end_idx = content.find(end_tag, start_idx)
    if end_idx == -1:
        raise ValueError(f"Could not find end tag in {svg_path}")

    new_ascii_block = f'<text x="{start_x}" y="0" class="ascii">\n' + '\n'.join(tspans) + '\n  '
    new_content = content[:start_idx] + new_ascii_block + content[end_idx:]
    Path(svg_path).write_text(new_content, encoding='utf-8')
    print(f"Updated {svg_path}")

if __name__ == '__main__':
    img_path = '/Users/surya/.gemini/antigravity-ide/brain/0fc06fc8-dac7-423f-b59d-714f24e91b7c/.user_uploaded/media_1787041629625.png'
    lines = generate_ascii(img_path, cols=110, rows=53)
    tspans = build_tspans(lines, start_x=22)

    Path('portrait.txt').write_text('\n'.join(lines), encoding='utf-8')
    Path('portrait_tspan.txt').write_text('\n'.join(tspans), encoding='utf-8')

    update_svg('dark.svg', tspans, start_x=22)
    update_svg('light.svg', tspans, start_x=22)
    print("Fine-tuned portrait width (110 cols) complete!")

import argparse
import io
import math
import os
import re
import sys

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def parse_marker_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    marker_size = None
    num_markers = None

    for line in lines:
        m = re.match(r"marker_size\s*=\s*(\d+)x(\d+)", line)
        if m:
            r = int(m.group(1))
            c = int(m.group(2))
            if r != c:
                raise ValueError('Only square markers are supported.')
            marker_size = r

        m = re.match(r"num_markers\s*=\s*(\d+)", line)
        if m:
            num_markers = int(m.group(1))

    if marker_size is None:
        raise ValueError('Could not parse marker_size from txt file.')

    markers = {}
    i = 0
    while i < len(lines):
        m = re.match(r"ID\s+(\d+):", lines[i])
        if m:
            marker_id = int(m.group(1))
            bits = []
            for j in range(1, marker_size + 1):
                if i + j >= len(lines):
                    raise ValueError(f'Not enough rows for marker ID {marker_id}')
                row_vals = lines[i + j].split()
                if len(row_vals) != marker_size:
                    raise ValueError(f'Marker ID {marker_id} row has wrong length.')
                row = [int(x) for x in row_vals]
                if any(x not in (0, 1) for x in row):
                    raise ValueError(f'Marker ID {marker_id} contains non-binary values.')
                bits.append(row)
            markers[marker_id] = np.array(bits, dtype=np.uint8)
            i += marker_size + 1
        else:
            i += 1

    ordered_ids = sorted(markers.keys())
    ordered_markers = [markers[mid] for mid in ordered_ids]

    if num_markers is not None and len(ordered_markers) != num_markers:
        print(
            f'Warning: txt says num_markers={num_markers}, '
            f'but parsed {len(ordered_markers)} marker patterns.',
            file=sys.stderr,
        )

    return marker_size, ordered_ids, ordered_markers


def get_byte_list_from_bits(bits):
    if hasattr(cv2.aruco.Dictionary, 'getByteListFromBits'):
        return cv2.aruco.Dictionary.getByteListFromBits(bits)
    return cv2.aruco.Dictionary_getByteListFromBits(bits)


def build_custom_dictionary(marker_size, marker_bits_list, maxcorr=1):
    rows = [get_byte_list_from_bits(bits) for bits in marker_bits_list]
    bytes_list = np.concatenate(rows, axis=0)
    return cv2.aruco.Dictionary(bytes_list, marker_size, maxcorr)


def make_marker_png_bytes(aruco_dict, marker_id, pixel_size=500, border_bits=1):
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        pixel_size,
        borderBits=border_bits,
    )
    ok, encoded = cv2.imencode('.png', marker_img)
    if not ok:
        raise RuntimeError(f'Could not encode marker {marker_id} as PNG.')
    return io.BytesIO(encoded.tobytes())


def count_groups(page_w_mm, page_h_mm, group_w_mm, group_h_mm, margin_mm, group_gap_mm):
    usable_w = page_w_mm - 2 * margin_mm
    usable_h = page_h_mm - 2 * margin_mm
    if usable_w < group_w_mm or usable_h < group_h_mm:
        return 0, 0

    cols = int((usable_w + group_gap_mm) // (group_w_mm + group_gap_mm))
    rows = int((usable_h + group_gap_mm) // (group_h_mm + group_gap_mm))
    return cols, rows


def choose_orientation(group_w_mm, group_h_mm, margin_mm, group_gap_mm):
    pw, ph = A4[0] / mm, A4[1] / mm

    p_cols, p_rows = count_groups(pw, ph, group_w_mm, group_h_mm, margin_mm, group_gap_mm)
    l_cols, l_rows = count_groups(ph, pw, group_w_mm, group_h_mm, margin_mm, group_gap_mm)

    p_total = p_cols * p_rows
    l_total = l_cols * l_rows

    if l_total > p_total:
        return 'landscape', landscape(A4), l_cols, l_rows
    return 'portrait', portrait(A4), p_cols, p_rows


def create_pdf(txt_path, pdf_path, marker_side_mm=2.0, cut_gap_mm=1.2, group_gap_mm=2.5,
               margin_mm=5.0, pages=1, border_bits=1, pixel_size=500):
    marker_size, ordered_ids, marker_bits = parse_marker_txt(txt_path)
    if len(ordered_ids) == 0:
        raise ValueError('No markers parsed from txt file.')

    aruco_dict = build_custom_dictionary(marker_size, marker_bits, maxcorr=1)

    marker_images = {
        marker_id: ImageReader(make_marker_png_bytes(aruco_dict, marker_id, pixel_size=pixel_size, border_bits=border_bits))
        for marker_id in ordered_ids
    }

    group_w_mm = len(ordered_ids) * marker_side_mm + (len(ordered_ids) - 1) * cut_gap_mm
    group_h_mm = marker_side_mm

    orientation_name, page_size, cols, rows = choose_orientation(
        group_w_mm, group_h_mm, margin_mm, group_gap_mm
    )

    if cols == 0 or rows == 0:
        raise ValueError('The chosen marker/group spacing does not fit on an A4 page.')

    page_w_pt, page_h_pt = page_size
    page_w_mm = page_w_pt / mm
    page_h_mm = page_h_pt / mm

    c = canvas.Canvas(pdf_path, pagesize=page_size)
    c.setTitle('Aruco Print Sheet')
    c.setAuthor('OpenAI')
    c.setSubject('Custom 3x3_5 ArUco marker print sheet')

    total_groups_per_page = cols * rows
    total_markers_per_page = total_groups_per_page * len(ordered_ids)

    left_margin_pt = margin_mm * mm
    top_margin_pt = margin_mm * mm
    marker_side_pt = marker_side_mm * mm
    cut_gap_pt = cut_gap_mm * mm
    group_gap_pt = group_gap_mm * mm
    group_w_pt = group_w_mm * mm
    group_h_pt = group_h_mm * mm

    usable_w_pt = page_w_pt - 2 * left_margin_pt
    usable_h_pt = page_h_pt - 2 * top_margin_pt
    used_w_pt = cols * group_w_pt + (cols - 1) * group_gap_pt
    used_h_pt = rows * group_h_pt + (rows - 1) * group_gap_pt

    x0 = left_margin_pt + (usable_w_pt - used_w_pt) / 2.0
    y_top = page_h_pt - top_margin_pt - (usable_h_pt - used_h_pt) / 2.0

    for _page in range(pages):
        for row in range(rows):
            for col in range(cols):
                gx = x0 + col * (group_w_pt + group_gap_pt)
                gy_top = y_top - row * (group_h_pt + group_gap_pt)
                gy = gy_top - group_h_pt

                for j, marker_id in enumerate(ordered_ids):
                    mx = gx + j * (marker_side_pt + cut_gap_pt)
                    c.drawImage(
                        marker_images[marker_id],
                        mx,
                        gy,
                        width=marker_side_pt,
                        height=marker_side_pt,
                        preserveAspectRatio=True,
                        mask='auto',
                    )
        if _page < pages - 1:
            c.showPage()

    c.save()

    return {
        'marker_size': marker_size,
        'num_unique_markers': len(ordered_ids),
        'orientation': orientation_name,
        'page_width_mm': page_w_mm,
        'page_height_mm': page_h_mm,
        'marker_side_mm': marker_side_mm,
        'cut_gap_mm': cut_gap_mm,
        'group_gap_mm': group_gap_mm,
        'margin_mm': margin_mm,
        'groups_per_page': total_groups_per_page,
        'markers_per_page': total_markers_per_page,
        'group_cols': cols,
        'group_rows': rows,
        'pages': pages,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate a dense printable PDF sheet from a custom ArUco txt file.')
    parser.add_argument('txt_path', help='Path to selected_markers.txt')
    parser.add_argument('pdf_path', nargs='?', default='aruco_print_sheet.pdf', help='Output PDF path')
    parser.add_argument('--marker-side-mm', type=float, default=2.0, help='Printed side length of each full ArUco marker')
    parser.add_argument('--cut-gap-mm', type=float, default=1.2, help='Gap between adjacent markers inside each 5-marker group')
    parser.add_argument('--group-gap-mm', type=float, default=2.5, help='Gap between neighboring 5-marker groups')
    parser.add_argument('--margin-mm', type=float, default=5.0, help='Page margin')
    parser.add_argument('--pages', type=int, default=1, help='Number of identical pages to generate')
    parser.add_argument('--pixel-size', type=int, default=500, help='Raster size used internally for each marker before placing into PDF')
    args = parser.parse_args()

    info = create_pdf(
        txt_path=args.txt_path,
        pdf_path=args.pdf_path,
        marker_side_mm=args.marker_side_mm,
        cut_gap_mm=args.cut_gap_mm,
        group_gap_mm=args.group_gap_mm,
        margin_mm=args.margin_mm,
        pages=args.pages,
        pixel_size=args.pixel_size,
    )

    print('PDF generated successfully.')
    for k, v in info.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()

def map_bbox_to_full_resolution(bbox, scale_factors, full_dims):
    x, y, w, h = bbox
    scale_x, scale_y = scale_factors
    full_width, full_height = full_dims
    x = float(x); y = float(y)
    w = float(w); h = float(h)
    scale_x = float(scale_x); scale_y = float(scale_y)
    x_full = max(0, int(round(x * scale_x)))
    y_full = max(0, int(round(y * scale_y)))
    w_full = max(1, int(round(w * scale_x)))
    h_full = max(1, int(round(h * scale_y)))
    x_full = min(x_full, full_width - 1)
    y_full = min(y_full, full_height - 1)
    w_full = min(w_full, full_width - x_full)
    h_full = min(h_full, full_height - y_full)
    return x_full, y_full, w_full, h_full

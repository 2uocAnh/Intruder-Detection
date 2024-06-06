import cv2

def get_roi(frame):
    r = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return r

def is_intersecting(bbox, roi):
    x, y, w, h = roi
    bx, by, bw, bh = bbox

    roi_x2, roi_y2 = x + w, y + h
    bbox_x2, bbox_y2 = bx + bw, by + bh

    intersecting = not (bx > roi_x2 or bbox_x2 < x or by > roi_y2 or bbox_y2 < y)
    print("WARNING!!!!")
    return intersecting

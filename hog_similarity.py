import cv2


def get_similarity(x1, x2):
    return -1 * cv2.CalcEMD2(
        x1,
        x2,
        cv2.CV_DIST_L2
    )

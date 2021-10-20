import cv2

def get_similarity(x1, x2):
    similarities = []
    for row in x2:
        similarities.append(1 / cv2.CalcEMD2(x1,row,cv2.CV_DIST_L2))
    return similarities

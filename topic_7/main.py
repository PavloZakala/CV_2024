import numpy as np
import cv2
import os

VIDEO_DATA_PATH = f"data/video.mp4"
IMAGES_DATA_PATH = f"data"

HARRIS_THRESHOLD = 0.08

def video_processing():

    cap = cv2.VideoCapture(VIDEO_DATA_PATH)
    i = 0
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # cv2.imwrite(os.path.join("data", f"{i}.png"), frame)
        # i += 1

        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        harris_corner_map = cv2.cornerHarris(gray, 2, 5, 0.04)

        frame[harris_corner_map > HARRIS_THRESHOLD * harris_corner_map.max()] = [0, 0, 255]

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def get_feature_points(image, harris_threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris_corner_map = cv2.cornerHarris(gray, 2, 5, 0.04)

    harris_corner_map[[0, 1]] = -1
    harris_corner_map[[-2, -1]] = -1

    harris_corner_map[:, [0, 1]] = -1
    harris_corner_map[:, [-2, -1]] = -1

    y, x = np.nonzero(harris_corner_map > harris_threshold * harris_corner_map.max())

    return np.column_stack((y, x))

def draw_matching(img1, img2, pts1, pts2, matches):
    pts1 = pts1.astype(np.uint)
    pts2 = pts2.astype(np.uint)

    kpts1 = [cv2.KeyPoint(int(pt[1]), int(pt[0]), 1) for pt in pts1]
    kpts2 = [cv2.KeyPoint(int(pt[1]), int(pt[0]), 1) for pt in pts2]

    img3 = cv2.drawMatches(img1, kpts1, img2, kpts2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matching simple", img3)

def simple_match_images(img1, img2):
    harris_threshold = 0.04

    fpts1 = get_feature_points(img1, harris_threshold)
    fpts2 = get_feature_points(img2, harris_threshold)

    kpts1 = [cv2.KeyPoint(int(pt[1]), int(pt[0]), 1) for pt in fpts1]
    kpts2 = [cv2.KeyPoint(int(pt[1]), int(pt[0]), 1) for pt in fpts2]

    # kimg1 = cv2.drawKeypoints(img1, kpts1, None)
    # cv2.imshow("Image1 keypoints", kimg1)
    #
    # kimg2 = cv2.drawKeypoints(img2, kpts2, None)
    # cv2.imshow("Image2 keypoints", kimg2)
    #
    # cv2.waitKey(0)

    def get_color_features(im, pts, scale_factor=0.5):
        features = []
        for _ in range(3):
            features += [
                im[pts[:, 0], pts[:, 1] - 2],
                im[pts[:, 0] - 1, pts[:, 1] - 1],
                im[pts[:, 0], pts[:, 1] - 1],
                im[pts[:, 0] + 1, pts[:, 1] - 1],
                im[pts[:, 0] - 2, pts[:, 1]],
                im[pts[:, 0] - 1, pts[:, 1]],
                im[pts[:, 0], pts[:, 1]],
                im[pts[:, 0] + 1, pts[:, 1]],
                im[pts[:, 0] + 2, pts[:, 1]],
                im[pts[:, 0] - 1, pts[:, 1] + 1],
                im[pts[:, 0], pts[:, 1] + 1],
                im[pts[:, 0] + 1, pts[:, 1] + 1],
                im[pts[:, 0], pts[:, 1] + 2]]

            im = cv2.resize(im, (int(im.shape[1] * scale_factor), int(im.shape[0] * scale_factor)))
            im = np.pad(im, ((2, 2), (2, 2), (0, 0)), 'edge')
            pts = ((pts + 2) * scale_factor).astype(np.int)


        return np.transpose(np.array(features), ((1, 0, 2))).reshape(-1, 3 * 13 * 3)


    print(f"Image 1 ", fpts1.shape[0])
    print(f"Image 2 ", fpts2.shape[0])

    features1 = get_color_features(img1, fpts1)
    features2 = get_color_features(img2, fpts2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(features1, features2)
    matches = sorted(matches, key=lambda x: x.distance)

    matches = matches[:70]

    sum = 0.0
    for m in matches:
        sum += np.abs(fpts1[m.queryIdx, 0] / img1.shape[0] - fpts2[m.trainIdx, 0] / img2.shape[0])

    res = sum / len(matches)

    print("Metric:", res)
    #
    draw_matching(img1, img2, fpts1, fpts2, matches)

    cv2.waitKey(0)

def sift_feature_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kpts, features = sift.detectAndCompute(gray, None)

    return kpts, features

def sift_matching_image(img1, img2):
    # https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html

    kpts1, features1 = sift_feature_points(img1)
    kpts2, features2 = sift_feature_points(img2)

    # kimg1 = cv2.drawKeypoints(img1, kpts1, None)
    # cv2.imshow("Image1 keypoints", kimg1)
    #
    # kimg2 = cv2.drawKeypoints(img2, kpts2, None)
    # cv2.imshow("Image2 keypoints", kimg2)
    #
    # cv2.waitKey(0)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(features1, features2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    good = good[:70]

    sum1 = 0.0
    for [m1] in good:
        sum1 += np.abs(kpts1[m1.queryIdx].pt[1] / img1.shape[0] - kpts2[m1.trainIdx].pt[1] / img2.shape[0])

    print("Metric:", sum1 / len(good))

    img3 = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, good, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matching sift", img3)

    # cv2.waitKey(0)


if __name__ == '__main__':
    image1 = cv2.imread(os.path.join(IMAGES_DATA_PATH, "0.png"))
    image2 = cv2.imread(os.path.join(IMAGES_DATA_PATH, "187.png"))

    # image1 = cv2.resize(image1, (image1.shape[1] // 4, image1.shape[0] // 4))
    # image2 = cv2.resize(image2, (image2.shape[1] // 5, image2.shape[0] // 5))
    #
    # image1 = cv2.resize(image1, (640, 480))
    # image2 = cv2.resize(image2, (640, 480))

    simple_match_images(image1, image2)
    sift_matching_image(image1, image2)


    cv2.waitKey(0)
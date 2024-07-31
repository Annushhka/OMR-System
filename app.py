from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import pandas as pd

# Directly set the image path
image_path = "omr_test_01.png"

# Define the answer key which maps the question number to the correct answer
df = pd.read_csv('answer.csv')
ANSWER_KEY = {}
for i in df.values:
    print(i[0], i[1])
    ANSWER_KEY[i[0]] = i[1]

print(ANSWER_KEY)

# Load the image, convert it to grayscale, blur it slightly, then find edges
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Find contours in the edge map, then initialize the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# Ensure that at least one contour was found
if len(cnts) > 0:
    # Sort the contours according to their size in descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # Loop over the sorted contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # If our approximated contour has four points, then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# Check if document contour was found
if docCnt is None:
    raise ValueError("Could not find document contour. Please check the input image.")

# Apply a four-point perspective transform to both the original image and grayscale image
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# Apply Otsu's thresholding method to binarize the warped piece of paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# Loop over the contours to find question contours
for c in cnts:
    # Compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # Label the contour as a question if it meets size and aspect ratio criteria
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# Sort the question contours top-to-bottom
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

# Loop over the questions in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # Sort the contours for the current question from left to right
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    # Loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # Construct a mask that reveals only the current bubble for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # Apply the mask to the thresholded image
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        # Check if this bubble has the most non-zero pixels so far
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # Determine if the bubbled answer is correct
    color = (0, 0, 255)
    k = ANSWER_KEY.get(q + 1, -1)  # Use get() to avoid KeyError
    if k == bubbled[1] + 1:
        color = (0, 255, 0)
        correct += 1
    # Draw the outline of the correct answer on the test
    cv2.drawContours(paper, [cnts[k - 1]], -1, color, 3)

# Calculate the score
score = (correct / len(ANSWER_KEY)) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)

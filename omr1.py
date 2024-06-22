import tkinter as tk
from tkinter import filedialog, Label, Frame, Canvas
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from PIL import Image, ImageTk



# Function to process the image and display the results
def process_image(image_path):
    # Define the answer key
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

    # Load the image, convert it to grayscale, blur it
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # Ensure that at least one contour was found
    if len(cnts) > 0:
        # Sort the contours according to their size
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Loop over the sorted contours
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # If approximated contour has four points, assume it's the paper
            if len(approx) == 4:
                docCnt = approx
                break

    # Apply perspective transform
    if docCnt is not None:
        paper = four_point_transform(image, docCnt.reshape(4, 2))
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find question contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)

        # Sort question contours
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        correct = 0

        # Loop over questions
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None

            # Loop over sorted contours
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            color = (0, 0, 255)
            k = ANSWER_KEY[q]

            if k == bubbled[1]:
                color = (0, 255, 0)
                correct += 1

            cv2.drawContours(paper, [cnts[k]], -1, color, 3)

        score = (correct / 5.0) * 100
        cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert image to RGB format for PIL
        paper_rgb = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
        # Convert image to ImageTk format
        paper_img = Image.fromarray(paper_rgb)
        paper_img_tk = ImageTk.PhotoImage(image=paper_img)

        # Update the image label
        image_label.configure(image=paper_img_tk)
        image_label.image = paper_img_tk
    else:
        print("No document found in the image.")


# Function to handle file selection and image processing
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)


# Create main application window
root = tk.Tk()
root.title("Exam Scoring")
root.geometry("800x650")

# Frame to hold file selection button
frame = Frame(root)
frame.pack(padx=20, pady=20)

# Create a button to select an image
select_button = tk.Button(frame, text="Select Image", command=select_image)
select_button.pack(side=tk.LEFT)

# Label to display selected image
image_label = Label(root)
image_label.pack(padx=20, pady=20)

# Run the Tkinter event loop
root.mainloop()

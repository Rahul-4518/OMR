import tkinter as tk
from tkinter import filedialog, Label, Frame, Text
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from PIL import Image, ImageTk

# Function to process the image and display the results
def process_image(image_path):
    global input_label, output_label, percentage_textbox, correct_textbox, wrong_textbox, total_textbox

    # Define the answer key
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

    # Load the image, convert it to grayscale, blur it
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Display the input image
    input_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_img_tk = ImageTk.PhotoImage(image=input_img)
    input_label.configure(image=input_img_tk)
    input_label.image = input_img_tk

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
        total_questions = 0

        # Loop over questions
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None
            total_questions += 1

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

        score = (correct / total_questions) * 100 if total_questions > 0 else 0.0
        cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert processed image to RGB format for PIL
        paper_rgb = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
        # Convert image to ImageTk format
        paper_img = Image.fromarray(paper_rgb)
        paper_img_tk = ImageTk.PhotoImage(image=paper_img)

        # Update the output label with the processed image
        output_label.configure(image=paper_img_tk)
        output_label.image = paper_img_tk

        # Update the textboxes
        percentage_textbox.delete(1.0, tk.END)  # Clear previous content
        percentage_textbox.insert(tk.END, "{:.2f}%".format(score))

        correct_textbox.delete(1.0, tk.END)
        correct_textbox.insert(tk.END, str(correct))

        wrong_textbox.delete(1.0, tk.END)
        wrong_textbox.insert(tk.END, str(total_questions - correct))

        total_textbox.delete(1.0, tk.END)
        total_textbox.insert(tk.END, str(total_questions))

    else:
        print("No document found in the image.")

# Function to handle file selection and image processing
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Update input label with selected image path
        input_label.configure(text="Selected Image: " + file_path)
        process_image(file_path)

# Create main application window
root = tk.Tk()
root.title("Exam Scoring")
root.geometry("1000x650")
background_color = ("light blue")

# Create frames
input_frame = Frame(root)
input_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.Y)

output_frame = Frame(root)
output_frame.pack(side=tk.LEFT, padx=20, pady=20)

# Input frame widgets
input_label = Label(input_frame, text="Input")
input_label.pack()

select_button = tk.Button(input_frame, text="Select Image", command=select_image)
select_button.pack()

# Display area for input image
input_image_label = Label(input_frame)
input_image_label.pack(padx=20, pady=20)

# Output frame widgets
output_label = Label(output_frame)
output_label.pack(padx=20, pady=20)

percentage_label = Label(output_frame, text="Percentage Score")
percentage_label.pack()

percentage_textbox = Text(output_frame, height=1, width=10)
percentage_textbox.pack()

correct_label = Label(output_frame, text="Correct Answers")
correct_label.pack()

correct_textbox = Text(output_frame, height=1, width=10)
correct_textbox.pack()

wrong_label = Label(output_frame, text="Wrong Answers")
wrong_label.pack()

wrong_textbox = Text(output_frame, height=1, width=10)
wrong_textbox.pack()

total_label = Label(output_frame, text="Total Questions")
total_label.pack()

total_textbox = Text(output_frame, height=1, width=10)
total_textbox.pack()

# Run the Tkinter event loop
root.mainloop()

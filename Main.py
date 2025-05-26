# IMAGE PROCESSING PROJECT GUI IMPLEMENTATION (Restored Layout)
# =========================================================
from tkinter import *
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os

# GLOBAL IMAGE VARIABLES
original_img = None
current_img = None
working_img = None

# WINDOW SETUP
win = Tk()
win.geometry('1100x650+50+20')
win.title('image processing project')
win.resizable(False, False)

# =========================================================
# DISPLAY AREA
f0 = Frame(width='400', height='650', bg='black')
f0.place(x=700, y=0)
display1 = Canvas(f0, width='350', height='200', bg='white')
display1.place(x=20, y=10)
display2 = Canvas(f0, width='350', height='200', bg='white')
display2.place(x=20, y=220)
display3 = Canvas(f0, width='350', height='200', bg='white')
display3.place(x=20, y=430)

# =========================================================
# UTILITY FUNCTIONS

def display_image(canvas, img):
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((350, 200))
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.image = img_tk
    canvas.create_image(0, 0, anchor=NW, image=img_tk)

def get_kernel():
    value = choose_kernel.get()
    if value:
        k = int(value[0])
        return (k, k)
    return (3, 3)

# =========================================================
# FUNCTIONALITY IMPLEMENTATIONS

def apply_filter_and_display(result):
    if working_img is not None:
        display_image(display1, working_img)
        display_image(display3, result)

def apply_low_pass():
    if working_img is None:
        return
    result = cv2.blur(working_img, get_kernel())
    apply_filter_and_display(result)

def apply_high_pass():
    if working_img is None:
        return
    blur = cv2.GaussianBlur(working_img, get_kernel(), 0)
    result = cv2.subtract(working_img, blur)
    apply_filter_and_display(result)

def apply_median():
    if working_img is None:
        return
    k = get_kernel()[0]
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    result = cv2.medianBlur(gray, k)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    apply_filter_and_display(result)

def apply_average():
    if working_img is None:
        return
    result = cv2.blur(working_img, get_kernel())
    apply_filter_and_display(result)

def apply_edge_detection(method):
    if working_img is None:
        return
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    if method == 'laplacian':
        result = cv2.Laplacian(gray, cv2.CV_64F)
    elif method == 'gaussian':
        result = cv2.GaussianBlur(gray, get_kernel(), 0)
    elif method == 'sobelx':
        result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    elif method == 'sobely':
        result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    elif method == 'canny':
        result = cv2.Canny(gray, 100, 200)
    else:
        return
    result = cv2.convertScaleAbs(result)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    apply_filter_and_display(result)

def apply_hough_lines():
    if working_img is None:
        return
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    result = working_img.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    apply_filter_and_display(result)

def apply_hough_circles():
    if working_img is None:
        return
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    result = working_img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
    apply_filter_and_display(result)

# =========================================================
# KERNEL DROPDOWN SETUP
kernel_size_label = Label(text='CHOOSE TYPE OF KERNEL', fg='white', bg='grey')
kernel_size_label.place(x=1010, y=490)
choose_kernel = ttk.Combobox(win, value=('3×3', '5×5', '7×7'), state='readonly')
choose_kernel.place(x=1010, y=510)
choose_kernel.set('3×3')

# =========================================================
# SAVE AND EXIT SECTION
f5 = Frame(width='700', height='40', bg='gray')
f5.place(x=0, y=610)

def save_image():
    global current_img
    if current_img is not None:
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, current_img)

def exit_app():
    win.destroy()

Button(f5, text='SAVE', fg='white', bg='black', width='20', command=save_image).place(x=30, y=0)
Button(f5, text='EXIT', fg='white', bg='black', width='20', command=exit_app).place(x=400, y=0)

# =========================================================
win.mainloop()

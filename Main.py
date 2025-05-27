from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


win = Tk()
win.geometry('1100x650+50+20')
win.title('image processing project')
win.resizable(False, False)

#GLOPAL VARIABLES
im1 = None
im2 = None
color_mode = StringVar(value="COLOR")

def show_image_on_canvas(img, canvas):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((350, 200))
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.image = img_tk
    canvas.create_image(0, 0, anchor=NW, image=img_tk)

def open_image_file():
    global im1 , im2
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG *.BMP")])
    if file_path :
        im1 = cv2.imread(file_path)
        im2 = im1
        show_image_on_canvas(im2 , display1)

def save_image_file():
    global im2
    if im2 is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, im2)

def exit_app():
    win.destroy()

def apply_color_mode():
    global im2
    if im1 is None:
        return
    if color_mode.get() == "GREY":
        gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        im2 = im1.copy()
    show_image_on_canvas(im1, display1)

def adjust_brightness(value=50):
    global im2
    if im2 is None:
        return
    brightness_value = value - 50  # slider value from 0 to 100, offset to -50 to +50
    img = cv2.convertScaleAbs(im2, alpha=1, beta=brightness_value)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(img, display2)

def adjust_contrast(value=50):
    global im2
    if im2 is None:
        return
    contrast_value = value / 50  # scale: 1.0 at center, 0.5 to 2.0 range
    img = cv2.convertScaleAbs(im2, alpha=contrast_value, beta=0)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(img, display2)

def show_histogram():
    global im2
    if im2 is None:
        return

    img = im2.copy()
    color = ('b', 'g', 'r')

    plt.figure("Histogram")
    if len(img.shape) == 2 or cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape == img.shape:
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
    else:
        # Color
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def equalize_histogram():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(eq_bgr, display2)

def get_kernel_size():
    return (5, 5)

def apply_low_pass():
    global im2
    if im2 is None:
        return
    result = cv2.blur(im2, get_kernel_size())
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(result, display3)

def apply_high_pass():
    global im2
    if im2 is None:
        return
    blur = cv2.GaussianBlur(im2, get_kernel_size(), 0)
    result = cv2.subtract(im2, blur)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(result, display3)

def apply_median_filter():
    global im2
    if im2 is None:
        return
    k = get_kernel_size()[0]
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, k)
    result = cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(result, display3)

def apply_average_filter():
    global im2
    if im2 is None:
        return
    result = cv2.blur(im2, get_kernel_size())
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(result, display3)

def get_gray_img():
    return cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

def edge_laplacian():
    gray = get_gray_img()
    result = cv2.Laplacian(gray, cv2.CV_64F)
    result = cv2.convertScaleAbs(result)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_gaussian():
    gray = get_gray_img()
    result = cv2.GaussianBlur(gray, (5, 5), 0)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_sobel_vert():
    gray = get_gray_img()
    result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    result = cv2.convertScaleAbs(result)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_sobel_horiz():
    gray = get_gray_img()
    result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    result = cv2.convertScaleAbs(result)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_prewitt_vert():
    gray = get_gray_img()
    kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    result = cv2.filter2D(gray, -1, kernel)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_prewitt_horiz():
    gray = get_gray_img()
    kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    result = cv2.filter2D(gray, -1, kernel)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_log():
    gray = get_gray_img()
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    result = cv2.Laplacian(blur, cv2.CV_64F)
    result = cv2.convertScaleAbs(result)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_canny():
    gray = get_gray_img()
    result = cv2.Canny(gray, 100, 200)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_zero_cross():
    gray = get_gray_img()
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    zero_cross = np.zeros_like(lap)
    zero_cross[(lap[:-1,:-1] * lap[1:,1:] < 0)] = 255
    result = cv2.convertScaleAbs(zero_cross)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_thicken():
    gray = get_gray_img()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    result = cv2.dilate(binary, kernel, iterations=1)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def edge_skeleton():
    gray = get_gray_img()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        open_ = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary, open_)
        eroded = cv2.erode(binary, element)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    show_image_on_canvas(cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR), display3)

def edge_thinning():
    gray = get_gray_img()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thinned = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary  # fallback if not available
    show_image_on_canvas(cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR), display3)
def detect_hough_lines():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Higher threshold to reduce excessive lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    result = im2.copy()

    if lines is not None:
        for i, (rho, theta) in enumerate(lines[:, 0]):
            if i >= 100:  # draw at most 100 lines
                break
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 1)

    show_image_on_canvas(im2, display1)
    show_image_on_canvas(result, display3)

def detect_hough_circles():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Tuned parameters for speed and clarity
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=35,
        minRadius=10,
        maxRadius=100
    )

    result = im2.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :100]:  # draw up to 100 circles
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

    show_image_on_canvas(im2, display1)
    show_image_on_canvas(result, display3)

def get_structuring_element():
    return cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def apply_dilation():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    kernel = get_structuring_element()
    result = cv2.dilate(gray, kernel, iterations=1)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def apply_erosion():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    kernel = get_structuring_element()
    result = cv2.erode(gray, kernel, iterations=1)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def apply_opening():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    kernel = get_structuring_element()
    result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

def apply_closing():
    global im2
    if im2 is None:
        return
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    kernel = get_structuring_element()
    result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    show_image_on_canvas(im2, display1)
    show_image_on_canvas(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), display3)

# DISPLAY AREA
f0 = Frame(width='400', height='650', bg='black')
f0.place(x=700, y=0)
display1 = Canvas(f0, width='350', height='200', bg='white')
display1.place(x=20, y=10)
display2 = Canvas(f0, width='350', height='200', bg='white')
display2.place(x=20, y=220)
display3 = Canvas(f0, width='350', height='200', bg='white')
display3.place(x=20, y=430)

# PREPARING
f1 = Frame(width='700', height='130', bg='gray')
f1.place(x=0, y=0)
l1 = Label(f1, text='LOAD IMAGE', fg='white', bg='grey', font='10')
l1.place(x=30, y=20)
open_image = Button(f1, text='open', fg='white', bg='black', width='10', font='8', command=open_image_file)
open_image.place(x=30, y=50)

l2 = Label(f1, text='COLOR MODE', fg='white', bg='grey', font='10')
l2.place(x=230, y=20)
color1 = Radiobutton(f1, text='COLOR', variable=color_mode, value='COLOR', fg='white', bg='grey', command=apply_color_mode)
color1.place(x=230, y=50)
color2 = Radiobutton(f1, text='GREY', variable=color_mode, value='GREY', fg='white', bg='grey', command=apply_color_mode)
color2.place(x=230, y=70)

# POINT TRANSFORMS
f2 = Frame(width='700', height='160', bg='gray')
f2.place(x=0, y=130)
l4 = Label(f2, text='POINT TRANSFORM', fg='white', bg='grey', font='8')
l4.place(x=30, y=10)
brightness = Button(f2, text='BRIGHTNESS ADJUSTMENT', fg='white', bg='black', width='50', command=lambda: adjust_brightness(70))
brightness.place(x=30, y=40)
contrast = Button(f2, text='CONTRAST ADJUSTMENT', fg='white', bg='black', width='50', command=lambda: adjust_contrast(70))
contrast.place(x=30, y=70)
histogram = Button(f2, text='HISTOGRAM', fg='white', bg='black', width='50', command=show_histogram)
histogram.place(x=30, y=100)
equalization = Button(f2, text='HISTOGRAM EQUALIZATION', fg='white', bg='black', width='50', command=equalize_histogram)
equalization.place(x=30, y=130)

# LOCAL TRANSFORMS
f3 = Frame(width='700', height='160', bg='gray')
f3.place(x=0, y=290)
l5 = Label(f3, text='LOCAL TRANSFORM', fg='white', bg='grey', font='8')
l5.place(x=30, y=10)
low = Button(f3, text='LOW PASS FILTER', command=apply_low_pass, fg='white', bg='black', width='25')
low.place(x=30, y=40)
high = Button(f3, text='HIGH PASS FILTER', command=apply_high_pass, fg='white', bg='black', width='25')
high.place(x=30, y=70)
median = Button(f3, text='MEDIAN FILTER(grey image)', command=apply_median_filter, fg='white', bg='black', width='25')
median.place(x=30, y=100)
average = Button(f3, text='AVERAGE FILTER', command=apply_average_filter, fg='white', bg='black', width='25')
average.place(x=30, y=130)

l6 = Label(f3, text='EDGE DETECTION FILTERS', fg='white', bg='grey', font='10')
l6.place(x=230, y=20)
edge1 = Radiobutton(f3, text='LAPLACIAN',command=edge_laplacian, fg='white', bg='grey')
edge1.place(x=230, y=50)
edge2 = Radiobutton(f3, text='GAUSSIAN',command=edge_gaussian, fg='white', bg='grey')
edge2.place(x=230, y=70)
edge3 = Radiobutton(f3, text='VERT SOBEL',command=edge_sobel_vert, fg='white', bg='grey')
edge3.place(x=230, y=90)
edge4 = Radiobutton(f3, text='HORIZ SOBEL',command=edge_sobel_horiz, fg='white', bg='grey')
edge4.place(x=230, y=110)
edge5 = Radiobutton(f3, text='VERT PREWITT',command=edge_prewitt_vert, fg='white', bg='grey')
edge5.place(x=330, y=50)
edge6 = Radiobutton(f3, text='HORIZ PREWITT',command=edge_prewitt_horiz, fg='white', bg='grey')
edge6.place(x=330, y=70)
edge7 = Radiobutton(f3, text='LAP OF GAU',command=edge_log, fg='white', bg='grey')
edge7.place(x=330, y=90)
edge8 = Radiobutton(f3, text='CANNY',command=edge_canny, fg='white', bg='grey')
edge8.place(x=330, y=110)
edge9 = Radiobutton(f3, text='ZERO CROSS',command=edge_zero_cross, fg='white', bg='grey')
edge9.place(x=430, y=50)
edge10 = Radiobutton(f3, text='THICKEN',command=edge_thicken, fg='white', bg='grey')
edge10.place(x=430, y=70)
edge11 = Radiobutton(f3, text='SKELETON',command=edge_skeleton, fg='white', bg='grey')
edge11.place(x=430, y=90)
edge12 = Radiobutton(f3, text='THINNING',command=edge_thinning, fg='white', bg='grey')
edge12.place(x=430, y=110)

# GLOBAL TRANSFORMS
f4 = Frame(width='700', height='160', bg='gray')
f4.place(x=0, y=450)
l6 = Label(f4, text='GLOBAL TRANSFORM', fg='white', bg='grey', font='8')
l6.place(x=30, y=10)
line_detect = Button(f4, text='LINE DETECTION USING HOUGH TRANSFORM', fg='white', bg='black', width='37', command=detect_hough_lines)
line_detect.place(x=30, y=50)
circle_detect = Button(f4, text='CIRCLE DETECTION USING HOUGH TRANSFORM', fg='white', bg='black', width='37', command=detect_hough_circles)
circle_detect.place(x=30, y=90)

l7 = Label(f4, text='MORPHOLOGICAL OP', fg='white', bg='grey', font='5')
l7.place(x=310, y=10)
dilation = Button(f4, text='DILATION', fg='white', bg='black', width='25', command=apply_dilation)
dilation.place(x=310, y=30)
erosion = Button(f4, text='EROSION', fg='white', bg='black', width='25', command=apply_erosion)
erosion.place(x=310, y=60)
opening = Button(f4, text='OPENING', fg='white', bg='black', width='25', command=apply_opening)
opening.place(x=310, y=90)
closing = Button(f4, text='CLOSING', fg='white', bg='black', width='25', command=apply_closing)
closing.place(x=310, y=120)

# SAVE AND EXIT
f5 = Frame(width='700', height='40', bg='gray')
f5.place(x=0, y=610)
save = Button(f5, text='SAVE', fg='white', bg='black', width='20', command=save_image_file)
save.place(x=30, y=0)
exit = Button(f5, text='EXIT', fg='white', bg='black', width='20', command=exit_app)
exit.place(x=400, y=0)

win.mainloop()

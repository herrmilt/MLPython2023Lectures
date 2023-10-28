import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os
import re

chars_to_move = {"q": -1, "w": 1, "a": -10, "s": 10, "z": -100, "x": 100, "c": -1000, "v": 1000}
class_colors = ["red", "blue", "green", "white", "black", "yellow", "brown", "gray"]


class BlurredRegion:
    def __init__(self, topx, topy, botx, boty):
        self.botx = botx
        self.topy = topy
        self.boty = boty
        self.topx = topx


class SelectedRegion:
    def __init__(self, class_idx, topx, topy, botx, boty):
        self.class_idx = class_idx
        self.botx = botx
        self.topy = topy
        self.boty = boty
        self.topx = topx


class MainWindow:

    def __init__(self, file_name, show_size=(1024, 768)):
        self.children = {}
        self.creating = False
        self.creating_blur = False
        self.moving_id = -1
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.rect_id = None
        self.start_x, self.start_y, self.original_rect = 0, 0, None

        self.window = tk.Tk()
        self.window.title("Annotation editor by MiltonGB.")

        self.cap = cv.VideoCapture(file_name)
        self.result_path = file_name + "_labeled"
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

        self.frame_idx = 0
        self.current_frame = None
        self.real_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.real_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.last_frame = None

        self.show_width, self.show_height = show_size

        self.window.geometry('%sx%s' % (int(self.show_width), int(self.show_height)))
        self.window.configure(background='grey')

        frame = tk.Frame(self.window, relief=tk.RAISED, borderwidth=1, height=60)
        frame.pack(side=tk.TOP, fill="x")

        tk.Label(frame, text="Current frame", padx=4).pack(side=tk.LEFT)
        self.frame_index_var = tk.IntVar(0)
        tk.Label(frame, textvariable=self.frame_index_var).pack(side=tk.LEFT)

        tk.Label(frame, text="Current class", padx=8).pack(side=tk.LEFT)
        self.current_class_idx_var = tk.IntVar(0)
        tk.Label(frame, textvariable=self.current_class_idx_var).pack(side=tk.LEFT)

        tk.Label(frame, text="Message: ", padx=8).pack(side=tk.LEFT)
        self.message_var = tk.StringVar()
        tk.Label(frame, textvariable=self.message_var).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.window, width=self.show_width, height=self.show_height,
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True)

        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.change_image(0)
        self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.topx, self.topy,
                                                    dash=(2, 2), fill='', width=3)
        self.change_current_class(0)

        self.canvas.bind('<Button-1>', self.get_mouse_posn)
        self.canvas.bind('<Button-3>', self.right_click)
        self.canvas.bind('<B1-Motion>', self.move_left_button)
        self.canvas.bind('<B3-Motion>', self.move_right_button)
        self.canvas.bind('<Button-2>', self.save_result)
        self.canvas.bind('<ButtonRelease-1>', self.mouse_release)
        self.canvas.bind('<ButtonRelease-3>', self.mouse_release_right)
        self.window.bind('<KeyPress>', self.key_pressed)
        self.window.bind('<Button-4>', self.go_prev_existing_img)
        self.window.bind('<Button-5>', self.go_next_existing_img)
        self.op_methods = {
            'Escape': lambda event: exit(0),
            "F2": self.save_result,
            "F8": self.clear_annotation_file,
            'F7': lambda event: self.remove_in_canvas(True, False),
            'F6': lambda event: self.remove_in_canvas(True, True)
        }

    def go_prev_existing_img(self, event):

        def filter_match(m):
            val = -1 if m is None else int(m.group(1))
            if val >= self.frame_idx:
                val = -1
            return val

        numbers = [filter_match(re.match("frame(\\d+).txt", f)) for f in os.listdir(self.result_path) if
                   os.path.isfile(os.path.join(self.result_path, f))]
        v = max(numbers)
        print(v)
        if v != -1:
            self.change_image(v)

    def go_next_existing_img(self, event):

        def filter_match(m):
            val = 10000000000 if m is None else int(m.group(1))
            if val <= self.frame_idx:
                val = 10000000000
            return val

        numbers = [filter_match(re.match("frame(\\d+).txt", f)) for f in os.listdir(self.result_path) if
                   os.path.isfile(os.path.join(self.result_path, f))]
        v = min(numbers)
        print(v)
        if v != 10000000000:
            self.change_image(v)

    def clear_annotation_file(self, event):
        image_file_name = os.path.join(self.result_path, f"frame{self.frame_idx}.jpg")
        txt_file_name = os.path.join(self.result_path, f"frame{self.frame_idx}.txt")
        if os.path.exists(image_file_name):
            os.remove(image_file_name)
            os.remove(txt_file_name)
            self.change_image(self.frame_idx)
        else:
            self.message_var.set("ERROR. Cannot remove annotation. Annotation do not exists.")

    def save_result(self, event):
        image_file_name = os.path.join(self.result_path, f"frame{self.frame_idx}.jpg")
        txt_file_name = os.path.join(self.result_path, f"frame{self.frame_idx}.txt")
        file = open(txt_file_name, "w")

        to_save = np.array(self.current_frame)
        for c_idx in self.children:
            c = self.children[c_idx]
            if isinstance(c, BlurredRegion):
                to_save[c.topy:c.boty, c.topx:c.botx] = \
                    cv.GaussianBlur(self.current_frame[c.topy:c.boty, c.topx:c.botx], (51, 51), 0)
                to_save[c.topy:c.boty, c.topx:c.botx] = \
                    cv.GaussianBlur(to_save[c.topy:c.boty, c.topx:c.botx], (51, 51), 0)
            elif isinstance(c, SelectedRegion):
                x, y, w, h = self.p1p2_to_yolo_box(c.topx, c.topy, c.botx, c.boty)
                file.write(f"{c.class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        file.close()
        cv.imwrite(image_file_name, to_save)
        self.message_var.set("Image and annotation saved!")

    def change_current_class(self, new_idx):
        self.current_class_idx_var.set(new_idx)
        self.canvas.itemconfig(self.rect_id, outline=class_colors[new_idx])

    def key_pressed(self, event):
        ch = str.lower(event.char)
        print("Pressed: " + event.keysym)
        if ch in chars_to_move:
            self.frame_idx += chars_to_move[ch]
            self.frame_idx = max(0, min(self.frame_idx, self.frame_count - 1))
            self.change_image(self.frame_idx)
        elif "0" <= ch <= "9":
            self.change_current_class(ord(ch) - ord("0"))
        elif event.keysym in self.op_methods:
            self.op_methods[event.keysym](event)

    def change_image(self, frame_idx):
        image_file_name = os.path.join(self.result_path, f"frame{frame_idx}.jpg")
        txt_file_name = os.path.join(self.result_path, f"frame{frame_idx}.txt")
        if os.path.exists(image_file_name) and os.path.exists(txt_file_name):
            self.current_frame = cv.imread(image_file_name)
            self.message_var.set("EXISTING annotation")
            # Load detections ...
            self.remove_in_canvas(True, True)
            with open(txt_file_name, "r") as file:
                for l in file.readlines():
                    comps = l.split()
                    p1x, p1y, p2x, p2y = self.yolo_box_to_p1p2(float(comps[1]), float(comps[2]), float(comps[3]),
                                                               float(comps[4]))
                    self.add_selected_region(int(comps[0]), p1x, p1y, p2x, p2y)
        else:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
            ret_val, self.current_frame = self.cap.read()
            self.message_var.set("Frame extracted from video")

        frame = cv.cvtColor(self.current_frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (self.show_width, self.show_height))
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.img = img
        self.canvas.itemconfig(self.image_id, image=img)
        self.frame_idx = frame_idx
        self.frame_index_var.set(self.frame_idx)
        self.last_frame = cv.cvtColor(self.current_frame, cv.COLOR_RGB2GRAY)

    def run(self):
        self.window.mainloop()

    def remove_element(self, idx):
        self.canvas.delete(idx)
        del self.children[idx]

    def right_click(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        if item[0] != 1:
            self.remove_element(item[0])
        else:
            self.creating_blur = True
            self.topx, self.topy = event.x, event.y

    def get_mouse_posn(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        self.creating = item[0] == 1
        if self.creating:
            self.topx, self.topy = event.x, event.y
        else:
            self.moving_id = item[0]
            self.start_x, self.start_y = event.x, event.y
            self.original_rect = self.canvas.coords(self.moving_id)

    def move_right_button(self, event):
        if self.creating_blur:
            self.botx, self.boty = event.x, event.y
            self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)

    def move_left_button(self, event):
        if self.creating:
            self.botx, self.boty = event.x, event.y
            self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)
        elif self.moving_id != -1:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            self.canvas.coords(self.moving_id,
                               self.original_rect[0] + dx,
                               self.original_rect[1] + dy,
                               self.original_rect[2] + dx,
                               self.original_rect[3] + dy)
            ch = self.children[self.moving_id]
            ch.topx = self.original_rect[0] + dx
            ch.topy = self.original_rect[1] + dy
            ch.botx = self.original_rect[2] + dx
            ch.boty = self.original_rect[3] + dy

    def add_selected_region(self, class_idx, topx, topy, botx, boty):
        r = SelectedRegion(class_idx,
                           min(topx, botx), min(topy, boty),
                           max(topx, botx), max(topy, boty))
        current_color = class_colors[self.current_class_idx_var.get()]
        new_rect = self.canvas.create_rectangle(topx, topy, botx, boty,
                                                fill=current_color, outline=current_color, stipple="gray25")
        self.children[new_rect] = r

    def change_selected_region(self, region_id, rect):
        obj = self.children[region_id]
        obj.topx = rect[0]
        obj.topy = rect[1]
        obj.botx = rect[0] + rect[2]
        obj.boty = rect[1] + rect[3]
        self.canvas.coords(region_id, obj.topx, obj.topy, obj.botx, obj.boty)

    def mouse_release(self, event):
        if self.creating:
            self.botx, self.boty = event.x, event.y
            if abs(self.botx - self.topx) < 20 or abs(self.boty - self.topy) < 20:
                return
            self.add_selected_region(self.current_class_idx_var.get(), self.topx, self.topy,
                                     self.botx, self.boty)
            self.canvas.coords(self.rect_id, 0, 0, 0, 0)
            self.creating = False

    def mouse_release_right(self, event):
        if self.creating_blur:
            self.botx, self.boty = event.x, event.y
            if abs(self.botx - self.topx) < 20 or abs(self.boty - self.topy) < 20:
                return
            r = BlurredRegion(min(self.topx, self.botx), min(self.topy, self.boty),
                              max(self.topx, self.botx), max(self.topy, self.boty))
            new_rect = self.canvas.create_rectangle(self.topx, self.topy, self.botx, self.boty,
                                                    fill='gray', outline='red')
            self.children[new_rect] = r
            self.canvas.coords(self.rect_id, 0, 0, 0, 0)
            self.creating_blur = False

    def remove_in_canvas(self, remove_detection=True, remove_blur=False):
        for c_idx in list(self.children):
            c = self.children[c_idx]
            if (isinstance(c, SelectedRegion) and remove_detection) or \
                    (isinstance(c, BlurredRegion) and remove_blur):
                self.canvas.delete(c_idx)
                del self.children[c_idx]

    def p1p2_to_yolo_box(self, topx, topy, botx, boty):
        width = botx - topx
        height = boty - topy
        return (topx + width / 2) * 1.0 / self.show_width, \
               (topy + height / 2) * 1.0 / self.show_height, \
               width * 1.0 / self.show_width, \
               height * 1.0 / self.show_height

    def yolo_box_to_p1p2(self, x, y, w, h):
        result_width = w * self.show_width
        result_height = h * self.show_height
        result_x = x * self.show_width - result_width / 2
        result_y = y * self.show_height - result_height / 2
        if result_x < 0:
            result_width += result_x
            result_x = 0
        if result_y < 0:
            result_height += result_y
            result_y = 0
        if result_x + result_width >= self.show_width:
            result_width = self.show_width - result_x
        if result_y + result_height >= self.show_height:
            result_height = self.show_height - result_y
        return int(result_x), int(result_y), int(result_x + result_width), int(result_y + result_height)


import sys

try:
    print(sys.argv)
    file_name = sys.argv[1]
    show_width = int(sys.argv[2])
    show_height = int(sys.argv[3])
    w = MainWindow(file_name, show_size=(show_width, show_height))
    w.run()
except Exception as exc:
    print("Correct usage: python3 annot_video.py file_name show_width show_height")
    print(exc)

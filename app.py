import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import re
from threading import Thread
import queue


class UserInterface:
    frame_box = queue.Queue(maxsize=1)
    cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    start_detecting = False
    how_much_train_img = 0
    count = 0
    
    def startDetectFaces(self):
        Thread(target=self.detectFaces, daemon=True).start()
    

    def EntryValidator(self, string, entry):
          if not string:
            entry.config(text="Please Give a num between 100, 200... 900")
            return False
          else:
            match = re.match(r"\d00", string)
            if match:
                entry.config(text="")
                self.start_detecting = True
                self.how_much_train_img = int(string)
                return True
            else:
                entry.config(text="Please Give a num between 100, 200... 900")
                return False


    def RecordFunction(self, live, label):
        res, frame = live.read()
        if res: 
              if self.start_detecting == True and self.count <= self.how_much_train_img:
                  self.count += 1
                  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                  faces = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                  for (x, y, w, h) in faces:
                      cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
              RGB_img = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
              FromArrImg = Image.fromarray(RGB_img)
              photo = ImageTk.PhotoImage(image=FromArrImg)
              label.photo_image = photo
              label.configure(image=photo)
              label.after(10, lambda: self.RecordFunction(live, label))
    
    def UserInterface(self):
        live = cv.VideoCapture(0)
        live.set(cv.CAP_PROP_FRAME_WIDTH, 500)
        live.set(cv.CAP_PROP_FRAME_HEIGHT, 400)
        live.set(cv.CAP_PROP_BUFFERSIZE, 1)
        root = tk.Tk()
        root.geometry("600x500")
        root.resizable(False, False)
        cameraView = tk.Label(root)
        cameraView.pack(padx=10, pady=10)
        self.RecordFunction(live, cameraView)
        input_field = tk.Entry(root)
        input_field.pack(anchor="center", padx=20, pady=10)
        btn = tk.Button(root, text="Click here to start learning", command=lambda: self.EntryValidator(input_field.get(), error))
        btn.pack(anchor="center", padx=20, pady=10)
        error = tk.Label(root, fg="#FF0000")
        error.pack(padx=10, pady=10)
        root.mainloop()



UI = UserInterface()
UI.UserInterface()
UI.startDetectFaces()
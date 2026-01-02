import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import os
import numpy as np
import re
from threading import Thread
import queue


class UserInterface:
    frame_box = queue.Queue(maxsize=1)
    cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    start_detecting = False
    how_much_train_img = 0
    count = 0
    imgs = []

    def createTrainModel(self, name):
        os.makedirs("Models", exist_ok=True)
        os.chdir("Models")
        recognizer = cv.face.LBPHFaceRecognizer_create()
        labels_map = {1: name}
        labels = []
        faces = []
        for face in self.imgs:
            faces.append(face)
            labels.append(1)
        recognizer.train(faces, np.array(labels))
        recognizer.save(f"{name}.yml")
    def EntryValidator(self, string, entry, btn, step):
          if step == "1":
            if not string:
                entry.config(text="Please Give a num between 100, 200... 900")
                return False
            else:
                match = re.match(r"\d00", string)
                if match:
                    entry.config(text="")
                    btn.config(bg="red", fg="white", state=tk.DISABLED)
                    self.start_detecting = True
                    self.how_much_train_img = int(string)
                    return True
                else:
                    entry.config(text="Please Give a num between 100, 200... 900")
                    return False
          elif step == "2":
              if not string:
                entry.config(text="Please Give Name with a-zA-Z0-9 and _")
                return False
              else:
                match = re.match(r"\w{2,25}", string)
                if match:
                    entry.config(text="Training Model...")
                    btn.config(bg="red", fg="white", state=tk.DISABLED)
                    self.createTrainModel(string)
                    entry.config(text="Model trained successfuly.")
                    return True
                else:
                    entry.config(text="Please Give Name with a-zA-Z0-9 and _")
                    return False


    def RecordFunction(self, live, label, count, btn, input_field):
        res, frame = live.read()
        if res: 
              if self.start_detecting == True:
                if self.count < self.how_much_train_img:
                    self.count += 1
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    faces = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            self.imgs.append(face)
                            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        count.config(text=f"{self.count}/{self.how_much_train_img} image captured.")
                else:
                    self.count = 0
                    self.start_detecting = False
                    count.config(text="Now enter the name of this person and click the button.")
                    btn.config(bg="green", fg="white", state=tk.ACTIVE, text="Click to train the model", command=lambda: self.EntryValidator(input_field.get(), count, btn, "2"))
                    
              RGB_img = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
              FromArrImg = Image.fromarray(RGB_img)
              photo = ImageTk.PhotoImage(image=FromArrImg)
              label.photo_image = photo
              label.configure(image=photo)
              label.after(10, lambda: self.RecordFunction(live, label, count, btn, input_field))
    
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
        count = tk.Label(root)
        count.pack(padx=10, pady=5)
        input_field = tk.Entry(root)
        input_field.pack(anchor="center", padx=20, pady=10)
        btn = tk.Button(root,bg="green", fg="white", text="Click here to start face capturing", command=lambda: self.EntryValidator(input_field.get(), count, btn, "1"))
        btn.pack(anchor="center", padx=20, pady=10)
        self.RecordFunction(live, cameraView, count, btn, input_field)
        root.mainloop()



UI = UserInterface()
UI.UserInterface()
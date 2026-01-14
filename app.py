import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2 as cv
import threading
from PIL import Image, ImageTk
import numpy as np
import re
import queue
import sys
import os

live = cv.VideoCapture(0)
live.set(cv.CAP_PROP_FRAME_WIDTH, 500)
live.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
os.makedirs("Models", exist_ok=True)
cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
if cascade.empty():
    messagebox.showerror(title="Error Message", message="Can't open the cascade, Try to use the full path!", icon='error')
    os._exit(1)
os.chdir('Models')
shared_frame = None
inTrain= False
inRecognition = False
running = True
Known = False
Box = threading.Lock()
how_much_pic = 0
count = 0

class Validator:
    @staticmethod
    def NumberVal(step, data, textLabel, function=None):
        global running, inTrain, how_much_pic
        if step == "1":
            if not inTrain:
                if not data:
                    textLabel.config(text="Enter a number from 100...900")
                else:
                    match = re.match(r'^[1-9][0-9]{2}$', data)
                    if match:
                        textLabel.config(text="")
                        inTrain = True
                        how_much_pic = int(data)
                    else:
                        textLabel.config(text="Enter a number from 100...900")
        elif step == "2":
            if not data:
                textLabel.config(text="Enter a name with a-z, A-Z, 0-9 and '_' .")
            else:
                match = re.match(r'^\w{2,25}$', data)
                if match:
                    textLabel.config(text="Training the model....")
                    if function != None:
                        threading.Thread(target=function, args=(data,textLabel)).start()
                else:
                    textLabel.config(text="Enter a name with a-z, A-Z, 0-9 and '_' .")


class CameraRecord:
    recognizer = cv.face.LBPHFaceRecognizer_create()
    coor = []
    faces = []
    predict = 0
    Face = []
    name = ""
    #Global Vars

    def FaceRecognition(self, model_name):
        global Known, inRecognition, running
        self.recognizer.read(model_name)
        while inRecognition and running:
            if len(self.Face) > 0:
                label, conf = self.recognizer.predict(self.Face)
                if conf <65:
                    Known = True
                    name = model_name.split(".")[0]
                    self.name = name
                else:
                    Known = False



    def LoadFaceRecognition(self, model_name, TextArea):
        global inRecognition
        if not os.path.exists(model_name):
            TextArea.config(text="Selected Model Not Found")
        else:
            inRecognition = True
            threading.Thread(target=self.FaceRecognition, args=(model_name,)).start()


    def TrainModel(self, name, label):
        ids = []
        for _ in range(len(self.faces)):
            ids.append(1)
        self.recognizer.train(self.faces, np.array(ids))
        self.recognizer.save(f"{name}.yml")
        label.config(text='Model Trained successfuly.')

    def TrainModel_Proc(self, btn, TextArea, Entry):
        TextArea.config(text="Give the name of this person.")
        btn.config(text="Click here to trian the model", command=lambda:Validator.NumberVal("2", Entry.get(), TextArea, self.TrainModel))


    def FaceDetecting(self, label, btn, Entry):
        global cascade, Box, shared_frame, inTrain, how_much_pic, count, inRecognition, Known, running
        while running:
            with Box:
                captured_frame = shared_frame
            if captured_frame is not None:
                captured_frame = cv.resize(captured_frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
                gray = cv.cvtColor(captured_frame, cv.COLOR_BGR2GRAY)
                self.coor = cascade.detectMultiScale(gray, 1.1, 6)
                for (x, y, w, h) in self.coor:
                    if inTrain:
                        if count < how_much_pic:
                            self.faces.append(gray[y:y+h, x:x+w])
                            count += 1
                            label.config(text=f"{count}/{how_much_pic} Captured.")
                        else:
                            inTrain = False
                            self.TrainModel_Proc(btn, label, Entry)
                    elif inRecognition:
                        self.Face = gray[y:y+h, x:x+w]

    def ShowRecord(self, CameraFieldOne, CameraFieldTwo, notebook):
        global running, cascade, Box, shared_frame, Known
        res, frame = live.read()
        if not running:
            return
        if res:
            with Box:
                shared_frame = frame
            if notebook.index(notebook.select()) == 0:
                CameraField = CameraFieldOne
                for (x, y, w, h) in self.coor:
                    cv.rectangle(frame, (x*2, y*2), (x*2 + w*2, y*2 + h*2), (0, 255, 0), 2)
            elif notebook.index(notebook.select()) == 1:
                CameraField = CameraFieldTwo
                for (x, y, w, h) in self.coor:
                    if Known:
                        cv.putText(frame, f"{self.name}", ((x-10)*2, (y-10)*2), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv.rectangle(frame, (x*2, y*2), (x*2 + w*2, y*2 + h*2), (0, 255, 0), 2)
                    else:
                        cv.putText(frame, "Unknown", ((x-10)*2, (y-10)*2), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv.rectangle(frame, (x*2, y*2), (x*2 + w*2, y*2 + h*2), (0, 0, 255), 2)
                    
            
            RGB_img = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
            photo = Image.fromarray(RGB_img)
            img = ImageTk.PhotoImage(photo)
            CameraField.photo = img
            CameraField.config(image=img)
        if running:
            CameraField.after(10, lambda: self.ShowRecord(CameraFieldOne, CameraFieldTwo, notebook))
    


class UserInterface:
    def __init__(self):
        root = tk.Tk()
        root.geometry('600x700')
        root.resizable(False, False)
        notebook = ttk.Notebook(root)
        notebook.pack(expand=True)
        Train_model = tk.Frame(notebook)
        Train_model.pack(expand=True, fill="both")
        Use_model = tk.Frame(notebook)
        Use_model.pack(expand=True, fill="both")
        notebook.add(Train_model, text="Train Model")
        notebook.add(Use_model, text='Use Model')

        #widgets of first tab
        CameraField = tk.Label(Train_model)
        CameraField.pack(pady=10)

        TextArea = tk.Label(Train_model, text="Welcome to face detector, here you can train you own model. Lets starting by\nentring the number of pictures you want to take from 100...900.")
        TextArea.pack(pady=15)

        TextField = tk.Entry(Train_model)
        TextField.pack(pady=10)

        StartBtn = tk.Button(Train_model, text="Click To Start", command=lambda: Validator.NumberVal("1", TextField.get(), TextArea))
        StartBtn.pack()

        #Widgets of second tab
        CameraFieldTwo = tk.Label(Use_model)
        CameraFieldTwo.pack(pady=10)

        CurentModels = [m for m in os.listdir() if m.endswith(".yml")]
        if len(CurentModels) > 0:
            ModelsList = ttk.Combobox(Use_model, values=CurentModels, state="readonly")
            ModelsList.set("Chose Model")
        else:
            ModelsList = ttk.Combobox(Use_model, state="readonly")
            ModelsList.set("No Model Trained")
        ModelsList.pack(padx=10, pady=5)
        def changeModelList():
            CurentModels = [m for m in os.listdir() if m.endswith(".yml")]
            if len(CurentModels) > 0:
                ModelsList.config(values=CurentModels)
                if ModelsList.get() not in CurentModels:
                    ModelsList.set("Chose Model")
            else:
                ModelsList.config(values=())
                ModelsList.set("No Model Trained")
            root.after(1000, changeModelList)

        changeModelList()

        Camera = CameraRecord()
        Camera.ShowRecord(CameraField, CameraFieldTwo, notebook)
        threading.Thread(target=Camera.FaceDetecting, args=(TextArea, StartBtn, TextField), daemon=True).start()
        TextAreaTwo = tk.Label(Use_model, text="Welcome to face detector, here you can train you own model. Lets starting by\nentring the number of pictures you want to take from 100...900.")
        TextAreaTwo.pack(pady=15)
        RecognitionBtn = tk.Button(Use_model, text="Click here", command=lambda:Camera.LoadFaceRecognition(ModelsList.get(), TextAreaTwo))
        RecognitionBtn.pack()



        def onClose():
            global running, live, inRecognition
            running = False
            inRecognition = False
            if live.isOpened():
                live.release()
            root.destroy()
            os._exit(0)

        root.protocol("WM_DELETE_WINDOW", onClose)
        root.mainloop()

App = UserInterface()
import tkinter.messagebox as mbox
from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image
from PIL import ImageTk

from DemoTensorFlow.DEEP1 import TestImage

sess,label_pred,predictions,image_input = TestImage.getSession()
labels = TestImage.getLabel()

class GUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("Face Recognition")

        menuBar = Menu(self.parent)
        self.parent.config(menu=menuBar)
        fileMenu = Menu(menuBar)
        fileMenu.add_command(label="Choose Image", command=self.chooseImage)
        menuBar.add_cascade(label="File", menu=fileMenu)

        self.labelImageFile = Label(self,text = "Choose Image")
        self.labelImageFile.pack()
        self.pack()


    def onExit(self):
        self.quit()

    def setGeometry(self):
        w = 300
        h = 450
        self.parent.geometry(("%dx%d+400+0") % (w, h))

    def chooseImage(self):
        filename = askopenfilename()
        print(filename)
        self.img = Image.open(filename)
        self.img = self.img.resize((300, 450), Image.ANTIALIAS)
        tatras = ImageTk.PhotoImage(self.img)
        self.labelImageFile.configure(image=tatras)
        self.labelImageFile.image = tatras
        self.setGeometry()
        pred = TestImage.getLabelImage(filename, sess, label_pred, predictions, image_input)
        label = labels[pred[0]]
        mbox.showinfo("Predictions", "Predictions :  "+str(label))


root = Tk()
ex = GUI(root)
ex.setGeometry()
root.mainloop()
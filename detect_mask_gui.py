import tkinter
from subprocess import call
from tkinter.ttk import Label, Frame
from PIL.ImageTk import PhotoImage

def detectVideo():
    call(["python", "detect_webcam.py"])
def exitt():
   exit()

#declare Tkinter
root=tkinter.Tk()

#set size
root.geometry('600x495')
frame = Frame(root, relief=tkinter.RIDGE, borderwidth=2)
frame.pack(fill=tkinter.BOTH, expand=1)

#set name
root.title('Mask Detector System Hanu')
label = Label(frame, text="Mask Detector System Hanu",font=('Times 35 bold'))
label.pack(side=tkinter.TOP)

#set background image
filename = PhotoImage(file="Background.png")
background_label = Label(frame,image=filename)
background_label.pack(side=tkinter.TOP)

#button with methods:
but1= tkinter.Button(frame, padx=5, pady=5, width=20,  fg='black', relief=tkinter.GROOVE, command=detectVideo, text='Detect Mask In Video', font=('helvetica 15 bold'))
but1.place(x=180,y=200)

but2= tkinter.Button(frame, padx=5, pady=5, width=20,  fg='black', relief=tkinter.GROOVE, command=exit, text='Exit', font=('helvetica 15 bold'))
but2.place(x=180,y=270)

#loop
root.mainloop()
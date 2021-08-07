#This code is used to create a GUI application for the project on DOA estimation. The GUI includes a main menu, a rotation controller page, a data collection page
#and the DOA estimation page. The main menu allows you to go to the other pages. The rotation controller page allows you to input the desired degree and sends
# the command to the arduino to rotate the automatic platform to that degree. The data collection page allows you to input the starting angle, the stopping angle,
# the angle resolution and the number of audio samples to record in every degree and proceeds to automatically record a .wav dataset and the .csv files with the 
# GCC data used in neural network training. The DOA estimation page is a demonstration of the DOA algorithm with a graph that indicates the DOA, a scale that alters 
# the energy threshold of the voice activity of detection, a choice of the 360 degree algorithm or the 60 degree and beamforming implementation and a tickbox 
# that allows the inclusion of the MUSIC algorithm.

#The object oriented approach was derived from sentdexs' tutorial in https://pythonprogramming.net/embedding-live-matplotlib-graph-tkinter-gui/
# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from tkinter import ttk
import Functions as fa
import scipy.io
import serial
import time
import os
import pandas as pd

matplotlib.use("TkAgg")

LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

pause = False


fs = 16000 #sampling rate kHz
duration = 0.15 #time frame duration in seconds

# Neural network models
path = 'lenet1.tflite'
path2 = 'FT10mires.tflite'

#Import beamforming weights for 60-degree implementation
mat = scipy.io.loadmat('beamformerwt1.mat')
mat = mat['wt']



def pyrec(fs,duration,channels =1):
    #Function that records multichannel 16 bit integer audio and returns a channelxchunksize numpy array

    #fs is the sampling rate
    #duration is the time frame duration of the recording in seconds
    #channels is the number of channels to record

    CHUNKSIZE = int(fs*duration) # fixed chunk size
    p = pyaudio.PyAudio()    
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate= fs, input=True, frames_per_buffer=CHUNKSIZE)

    data = stream.read(CHUNKSIZE)
    numpydata = np.frombuffer(data, dtype=np.int16)
    numpydata = np.reshape(numpydata, (CHUNKSIZE, channels))
    numpydata = [numpydata[:,i] for i in range(channels)]
    
    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return numpydata

def _quit():
    app.quit()     # stops mainloop
    app.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

def PointsInCircum(r,n=100):
    #returns the points that are used to visualize DOA in a circle
    pi = np.pi
    return [(np.cos(2*pi/n*x)*r,np.sin(2*pi/n*x)*r) for x in range(0,n+1)]
   
            

class Demo(tk.Tk):
    #Class that displays the GUI pages
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Demo Dev Stage")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        #To add more pages create a page class with the tkinter content and add it in the tuple below
        for F in (StartPage, PageOne, PageTwo, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        #Function that raises the different Pages
        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):
    #Main menu page, has buttons that lead to the rest pages
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Welcome to the Main Menu", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Rotation Controller Page",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Data Collection Page",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="DOA Estimation Page",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()
        
        buttonquit = ttk.Button(self , text = "Quit", command = _quit )
        
        buttonquit.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):

        #Exception checks if an Arduino is serially connected
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 57600, timeout=1)
            self.ser.flush()
            self.message = self.ser.readline().decode('utf-8').rstrip(),
        except:
            self.message = 'Connect the Respeaker to the Arduino and restart the GUI'



        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Rotary stage control", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        btngraph = ttk.Button(self, text="To graph page",
                            command=lambda: controller.show_frame(PageThree))
        btngraph.pack()

        
        lblen = tk.Label(self, text="Enter the rotating angle", font=LARGE_FONT)
        lblen.pack(pady=10,padx=10)

        self.entry = tk.Entry(self)
        self.entry.pack(pady=10,padx=10)
        
        btnrot = ttk.Button(self, text = "Rotate",
                            command= self.rotate)
        btnrot.pack()

        self.lblcurrent = tk.Label(self, text= self.message, font=LARGE_FONT)
        self.lblcurrent.pack(pady=10,padx=10)
        

    def rotate(self):
        #Sends command to the arduino that turns the stepper motor to the desired degree
        angle = self.entry.get()
        string = 'SRA ' + angle + "\r\n" #add \r\n bc arduino commands end with carriage return and newline
        self.ser.write(string.encode('utf-8'))
        self.ser.write(b'GRA\r\n') #request from the arduino the degree to ensure proper command transmission
        line = self.ser.readline().decode('utf-8').rstrip() 
        self.lblcurrent['text'] = 'The current angle is ' + line


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Data collection page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        startframe = tk.Frame(self)

        self.lblstart = tk.Label(startframe, text = 'Starting angle:')
        self.lblstart.pack(pady=10,padx=10, side = tk.LEFT)

        self.entstart = tk.Entry(startframe)
        self.entstart.pack(pady=10,padx=10, side = tk.RIGHT)

        startframe.pack()
        
        
        stopframe = tk.Frame(self)

        self.lblstop = tk.Label(stopframe, text = 'Stopping angle:')
        self.lblstop.pack(pady=10,padx=10, side = tk.LEFT)

        self.entstop = tk.Entry(stopframe)
        self.entstop.pack(pady=10,padx=10, side = tk.RIGHT)

        stopframe.pack()
        
        
        resframe = tk.Frame(self)

        self.lblres = tk.Label(resframe, text = 'Insert resolution in degrees:')
        self.lblres.pack(pady=10,padx=10, side = tk.LEFT)

        self.entres = tk.Entry(resframe)
        self.entres.pack(pady=10,padx=10, side = tk.RIGHT)

        resframe.pack()

        samframe = tk.Frame(self)

        self.lblsam = tk.Label(samframe, text = 'Insert number of recordings per degree:')
        self.lblsam.pack(pady=10,padx=10, side = tk.LEFT)

        self.entsam = tk.Entry(samframe)
        self.entsam.pack(pady=10,padx=10, side = tk.RIGHT)

        samframe.pack()

        btnstart = ttk.Button(self, text="Start Recording",
                            command= self.record)
        btnstart.pack()

        self.lblstat = tk.Label(self, text= '', font=LARGE_FONT)
        self.lblstat.pack(pady=10,padx=10)

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 57600, timeout=1)
            self.ser.flush()
            self.message = self.ser.readline().decode('utf-8').rstrip(),
        except:
            self.lblstat['text'] = 'Connect the Respeaker to the Arduino and restart the GUI'

    def record(self):
        #Function that reads starting/stopping degree, resolution and number of samples and automatically records them
        try:
            start = int(self.entstart.get())
            stop = int(self.entstop.get())
            res = int(self.entres.get())
            samples = int(self.entsam.get())
        except ValueError:
            self.lblstat["text"] = 'Please insert integer numbers'
        
        rotangles = np.arange(start,stop+res,res)
        tim = time.ctime(time.time())

        #creates the directories where the GCC .csv files, .wav audio recordings and a readme file with some metadata are saved 
        directoryname = '/home/respeaker/Recordings/Recording' + str(tim)
        mode = 0o777
        os.mkdir(directoryname, mode)
        recdir = os.path.join(directoryname, 'Recording')
        os.mkdir(recdir,mode)
        gccdir = os.path.join(directoryname, 'GCC')
        os.mkdir(gccdir,mode)

        f = open( directoryname + '/Readme.txt' , 'w+')
        f.write('Sampling rate is ' +str(fs))
        f.write('Audio frames length in seconds is ' +str(duration)+'\n')
        f.write('Starting angle is ' +str(start)+'\n')
        f.write('Stopping angle is ' +str(stop)+'\n')
        f.write('Resolution in degrees is ' +str(res)+'\n')
        f.write('Number of samples per degree is ' +str(samples)+'\n')
        f.close()


        for i in rotangles:
            
            df = pd.DataFrame()

            j = 0


            string = 'SRA ' + str(i) + "\r\n"
            self.ser.write(string.encode('utf-8'))


            self.lblstat["text"] = 'Recording is ' + str((i*100)/len(rotangles))+ ' % complete'

            while j<samples:

                filepath = recdir+'/degree'+str(i)+ 'recording' + str(j)
                #records voiced data and saves them
                recording, cond = fa.recnsaveWAV(filepath, fs,duration)
                
                if cond:
                    recording = fa.order(recording)
                    X,Y,GCC = fa.myinwhole(recording,fs)
                    df = df.append(pd.DataFrame(GCC))
                    j += 1
            df.to_csv(gccdir + '/gcc' + str(i)+'.csv')
        self.lblstat["text"] = 'Recording  completed'


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        
        self.first = 0
        self.th = 1500
        self.algsel = True
        self.music = tk.BooleanVar(value=False)
        
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='''Direction of arrival estimation page, 
            Click anywhere on the GUI to Pause/Unpause''', font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        
        self.lblp = tk.Label(self, text = 'Paused', font = LARGE_FONT)
        self.lblp.pack()

        scale = tk.Scale(self, orient='horizontal', from_=0, to=1000, label = 'VAD Threshold (Recommended 500)' , command = self.thres)
        scale.pack()

        self.lbldoa = tk.Label(self, text='The DOA is :', font=LARGE_FONT)
        self.lbldoa.pack(pady=10,padx=10)

        btnhome = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        btnhome.pack()

        btnrot = ttk.Button(self, text="To rotation controller",
                            command=lambda: controller.show_frame(PageOne))
        btnrot.pack()

        
        self.algselect = tk.Frame(self) 
        self.btn360 = tk.Button(self.algselect, text = '360 algorithm' , fg = 'blue', command = self.algsw360)
        self.btnbeam = tk.Button(self.algselect, text = '60 degree/ beamforming', command = self.algswbeam)

        self.btn360.pack( side = tk.LEFT)
        self.btnbeam.pack( side = tk.RIGHT)

        check = ttk.Checkbutton(self.algselect, text='MUSIC', 
                        variable = self.music)
        check.pack( side = tk.RIGHT)

        self.algselect.pack()


        self.f = Figure(figsize=(5,5), dpi=100)
        self.cf = self.f.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.f.canvas.mpl_connect('button_press_event', self.onClick)
        self.ani = animation.FuncAnimation(self.f, self.animate, interval=10)
        
    #Functions that switch between the DOA algorithms
    def algsw360(self):
        self.algsel = True
        self.btn360['fg'] = 'blue'
        self.btnbeam['fg'] = 'black'
        
    def algswbeam(self):
        self.algsel = False
        self.btn360['fg'] = 'black'
        self.btnbeam['fg'] = 'blue'
        

    #Reads VAD threshold slider
    def thres(self, val):
        self.th = int(val) 

    #Animate function that enables real time DOA estimation and plotting
    def animate(self,i):
        self.data = pyrec(fs , duration , 6)

        #VAD with energy threshold
        if np.sum(np.square(self.data[0]))/len(self.data[0]) > self.th and np.sum(np.square(self.data[4]))/len(self.data[0]) > self.th and np.sum(np.square(self.data[2]))/len(self.data[0]) > self.th:
            if self.algsel:
                #360 degree DOA estimation algorithm
                self.data = fa.order(self.data)
                self.newind = fa.mic_index(0)
                self.X,self.Y,self.GCC = fa.myinwhole(self.data,fs,self.newind)
                self.GCC = np.array([self.GCC], dtype = np.float32)
                self.GCC = np.expand_dims(self.GCC,3)
                self.nnres = np.argmax(fa.setinterpreter(path,self.GCC))
                self.doa = fa.doaconv360(0,self.nnres)
                
            
            else:
                #60- degree w beamforming DOA estimation algorithm
                self.data = fa.order(self.data)
                self.y = [fa.beamform(self.data,i,mat) for i in range(6)]
                self.imax = fa.power(self.y,fs)
                self.newind = fa.mic_index(self.imax)
                self.X,self.Y,self.GCC = fa.myinwhole(self.data,fs,self.newind)
                self.GCC = np.array([self.GCC], dtype = np.float32)
                self.nnres = np.argmax(fa.setinterpreter(path2,self.GCC))
                self.doa = fa.doaconv(self.imax,self.nnres)
                
            #DOA plotting function
            self.lbldoa["text"] = "The DOA is : " + str(self.doa)
            self.cf.clear()
            self.cf.set_aspect('equal','datalim')
            self.cf.plot(c[:,0],c[:,1])
            self.cf.plot(c[self.doa,0],c[self.doa,1],'r*')
            
            #option to include MUSIC algorithm
            if self.music.get():
                musdoa = fa.music(self.data)
                self.cf.plot(c[musdoa,0],c[musdoa,1],'g*')

        else:
            self.lbldoa["text"] = "No voice detected"
            self.cf.clear()
            self.cf.set_aspect('equal','datalim')
            self.cf.plot(c[:,0],c[:,1])
        #This part makes the animate function to stop at the start of the GUI
        if self.first < 2:
            self.ani.event_source.stop()
            self.first += 1

    #This function pauses the animation by clicking inside the GUI
    def onClick(self,event):
        global pause
        if pause:
            self.ani.event_source.stop()
            self.lblp["text"] = 'Paused'
            pause = False
        else:
            self.ani.event_source.start()
            self.lblp["text"] = 'Resumed'
            pause = True

#main
if __name__ == "__main__":

    c = np.array(PointsInCircum(1,360))
    app = Demo()
    app.mainloop()

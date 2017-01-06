import itertools
import pickle
import datetime
import os
import numpy as np

def log2lin(x,xlim):
    b = np.log10(xlim[1]/xlim[0])/(xlim[1]-xlim[0])
    a = xlim[1]/10**(b*xlim[1])
    x = a*10**(b*x)
    return x
    
def stripfile(file):
    file = file.replace('.png','')  # strip file type
    file = file.replace('.jpg','')
    return file

class sample_plot(object):
    
    def __init__(self, data, x_origin, y_origin, x_ref, y_ref,
                 x_fig, y_fig, ax_eq, ax, fig, path, file,
                 xscale='linear',yscale='linear'):
        self.xscale = xscale
        self.yscale = yscale
        self.cord_flag = 0
        self.count_data = itertools.count(1)
        self.data = data
        self.path = path
        self.file = file
        self.set_folder()
        self.x = []
        self.y = []
        self.xy = []
        self.x_fig = x_fig
        self.y_fig = y_fig
        self.ax_eq = ax_eq
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.ax = ax
        self.fig = fig
        self.cycle_click = itertools.cycle([1,2])
        self.cycle_axis = itertools.cycle([1,2,3,4])
        self.count_click = next(self.cycle_click)
        self.count_axis = next(self.cycle_axis)
        self.cid = x_origin.figure.canvas.mpl_connect('button_press_event', self)
        if self.ax_eq == -1:
            self.count_axis = next(self.cycle_axis)
            self.count_axis = next(self.cycle_axis)
        
    def set_folder(self):
        self.file = stripfile(self.file)
        date_str = datetime.date.today().strftime('%Y_%m_%d')  # today
        self.folder = self.path+'imagedata/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.file = date_str+'_'+self.file+'.pkl'

    def __call__(self, event):
        if event.button == 1:  # select button
            if self.count_click == 1:  # enter axis
                if self.count_axis == 1:
                    self.x_origin.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select x referance (x='+str(self.x_fig[1])+')')
                    self.x_origin.figure.canvas.draw()
                elif self.count_axis == 2:
                    self.x_ref.set_data(event.xdata, event.ydata)
                    if self.ax_eq == 1:
                        self.ax.set_title('select data (select|finish|undo)')
                        self.count_click = next(self.cycle_click)
                    else:
                        self.ax.set_title('select y-origin (y='+str(self.y_fig[0])+')')   
                    self.x_ref.figure.canvas.draw()                    
                elif self.count_axis == 3:
                    self.y_origin.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select y referance (y='+str(self.y_fig[1])+')')
                    self.y_origin.figure.canvas.draw()
                else:
                    self.y_ref.set_data(event.xdata, event.ydata)
                    self.ax.set_title('select data (select|finish|undo)')
                    self.y_ref.figure.canvas.draw()
                    self.count_click = next(self.cycle_click)
                self.count_axis = next(self.cycle_axis)
            else:  # enter data
                self.x.append(event.xdata)
                self.y.append(event.ydata)
                self.data.set_data(self.x, self.y)
                self.y_ref.figure.canvas.draw()    
        if event.button == 2:  # enter button
            if self.count_click == 2:  # exit and save
                if len(self.x) == 0:
                    self.fig.canvas.mpl_disconnect(self.cid)
                else:
                    if self.cord_flag == 0:
                        self.set_cord()
                        self.cord_flag = 1
                    self.store_data()
                    self.x, self.y = [], []  # reset
                    self.data.set_data(self.x,self.y)
                    self.data.figure.canvas.draw()
        if event.button == 3:  # remove data points
            if self.count_click == 2: 
                if len(self.x) > 0:
                    self.x.pop(len(self.x)-1)
                if len(self.y) > 0:    
                    self.y.pop(len(self.y)-1)
                self.data.set_data(self.x, self.y)
                self.data.figure.canvas.draw() 
                
    def set_cord(self):
        if self.ax_eq == 1:  # referance from x-dir
            x_ref = self.x_ref.get_xydata()[0][0]
            self.x_o = self.x_origin.get_xydata()[0]
            self.x_o[1] = -self.x_o[1]
            self.y_o = self.x_o
            self.x_scale = (self.x_fig[1]-self.x_fig[0])/(x_ref-self.x_o[0])
            self.y_scale = self.x_scale
        elif self.ax_eq == -1:  # referance from y-dir
            y_ref = -self.y_ref.get_xydata()[0][1]
            self.y_o = self.y_origin.get_xydata()[0]
            self.y_o[1] = -self.y_o[1]
            self.x_o = self.y_o
            self.y_scale = (self.y_fig[1]-self.y_fig[0])/(y_ref-self.y_o[1])
            self.x_scale = self.y_scale
        else:  # referance from x and y
            x_ref = self.x_ref.get_xydata()[0][0]
            y_ref = -self.y_ref.get_xydata()[0][1]
            self.x_o = self.x_origin.get_xydata()[0]
            self.x_o[1] = -self.x_o[1]
            self.y_o = self.y_origin.get_xydata()[0]
            self.y_o[1] = -self.y_o[1]
            self.x_scale = (self.x_fig[1]-self.x_fig[0])/(x_ref-self.x_o[0])
            self.y_scale = (self.y_fig[1]-self.y_fig[0])/(y_ref-self.y_o[1])
            
    def store_data(self):
        data = self.data.get_xydata()
        x = data[:,0]
        y = -data[:,1]
        x = self.x_scale*(x-self.x_o[0])+self.x_fig[0]
        y = self.y_scale*(y-self.y_o[1])+self.y_fig[0]
        if self.xscale == 'log':
            x = log2lin(x,self.x_fig)
        if self.yscale == 'log':
            y = log2lin(y,self.y_fig)
        points = {'x':x,'y':y}
        limits = {}
        for var in ['x_o','x_scale','x_fig','y_o','y_scale','y_fig']:
            limits[var] = getattr(self,var)
        with open(self.folder+self.file, 'wb') as output:
                pickle.dump(points,output,-1)
                pickle.dump(data,output,-1)
                pickle.dump(limits,output,-1)
 
def data_mine(path,file,**kw):
    from matplotlib import pyplot as plt
    from amigo.png_tools import sample_plot
    import matplotlib.image as mpimg
    label = kw.get('label','')
    if 'scale' in kw:
        scale = kw.get('scale')
        xscale = scale
        yscale = scale
    else:
        if 'xscale' in kw:
            xscale = kw.get('scale')
        else:
            xscale = 'linear'
        if 'yscale' in kw:
            yscale = kw.get('scale')
        else:
            yscale = 'linear'
    x_fig = kw['xlim']
    y_fig = kw['ylim']
    ax_eq = 0
    
    if len(x_fig) == 0:
        ax_eq = -1
        x_fig = y_fig
        
    if len(y_fig) == 0:
        ax_eq = 1
        y_fig = x_fig

    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111)
    
    origin = 'upper'
    if '.png' not in file and '.jpg' not in file:
        file += '.png'
    image = mpimg.imread(path+file)
    ax.imshow(image, origin=origin)
    if ax_eq == -1:
        ax.set_title('select y-origin (y='+str(y_fig[0])+')')        
    else:
        ax.set_title('select x-origin (x='+str(x_fig[0])+')')
    
    #default markers
    data, = ax.plot([0],[0],'gs-')
    x_origin, = ax.plot([0],[0],'ro') 
    y_origin, = ax.plot([0],[0],'bo') 
    x_ref, = ax.plot([0],[0],'rx') 
    y_ref, = ax.plot([0],[0],'bx')

    sample_plot = sample_plot(data,x_origin,y_origin,x_ref,y_ref,x_fig,y_fig,
                              ax_eq,ax,fig,path,file+'_'+label,
                              xscale=xscale,yscale=yscale)

def data_load(path,file,**kwargs):
    date = kwargs.get('date',datetime.date.today().strftime('%Y_%m_%d'))
    label = kwargs.get('label','')
    file = stripfile(file)
    with open(path+'imagedata/'+date+'_'+file+'_'+label+'.pkl', 'rb') as input:
                points = pickle.load(input)
                #data = pickle.load(input)
                #limits = pickle.load(input)
    return points






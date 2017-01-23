import numpy as np

def OOB(var, delta=1e-3):  # out of bounds
    var = np.append(var[0]-delta, var)
    var = np.append(var, var[-1]+delta)
    return (var,len(var))
    
def edge(index, Narray):
    on_edge = False    
    for i,N in zip(index,Narray):
        if i==0 or i==N-1:
            on_edge = True
    return on_edge

class table(object):
    def __init__(self,filename,ext='.lib'):
        f = open(filename+ext, 'w')         
        self.f = f
        
    def nop(self):
        self.f.write('/nop\n')  # supress output
        
    def go(self):
        self.f.write('/go\n')  # resume output
        
    def load(self, name, data, index=[]):
        if len(data) == 1:
            data = np.array([data[0],data[0]])
            index = [np.array([index[0][0]-1e-34,index[0][0]+1e-34])]
        self.name = name
        self.data = data
        self.index = index
        self.shape = np.shape(data)
        self.nD = len(self.shape)  # 1D,2D,...5D

    def write(self,primary,csysid=''):
        self.write_header(primary,csysid=csysid)
        self.write_index()
        self.write_data()
        #self.f.write('\n')
        
    def write_array(self):
        self.write_header([])
        self.write_data()
        #self.f.write('\n')
        
    def write_text(self, text):
        self.f.write(text+'\n')
        #self.f.write('\n')
        
    def close(self):
        self.f.close()
        
    def boundary(self, BC, arguments, surface):
        table = ''
        for arg in arguments:
            table=table+',%'+arg+'%'
        table = table[1:]
        if BC is 'HGEN':
            self.f.write('bfe,'+surface+','+BC+',1,'+table+'\n\n')    
        elif BC is 'TCC':
            self.f.write('RMODIF,'+surface+',14,'+table+'\n\n')
        else:
            if BC in ['TEMP']:
                BCtype = 'd'
            elif BC in ['CONV','HFLUX']:
                BCtype = 'sf'
            self.f.write(BCtype+','+surface+','+BC+','+table+'\n\n')
        
    def link_args(self, independant):
        for i,var in enumerate(independant):
            self.f.write('*DIM,'+var+',TABLE,1,,,TIME\n')
            self.f.write(var+'(1,1) = arg'+str(i+1)+'\n\n')

    def declare(self,name,shape):
        self.name = name
        self.shape = shape
        self.nD = len(self.shape)
        self.write_header([])
        
    def write_header(self,primary,csysid=''):
        var_list = ['','','',csysid]
        var_str = ''
        for i,var in enumerate(primary):  # primary variables
            var_list[i] = var
        for var in var_list:
            var_str = var_str+','+var
            
        if not primary:  # empty primary array (array)
            array_type = 'ARRAY'
        else:
            array_type = 'TABLE'
            
        self.ijk = np.ones(5)  # upto 5D
        for D in range(self.nD):
            self.ijk[D] = int(self.shape[D])
        if self.nD <= 3:
            self.f.write('*DIM,'+self.name+','+array_type\
                    +',{:1.0f}'.format(self.ijk[0])\
                    +',{:1.0f}'.format(self.ijk[1])\
                    +',{:1.0f}'.format(self.ijk[2])\
                    +var_str+'\n') 
        elif self.nD == 4:
            self.f.write('*DIM,'+self.name+','+array_type[:3]+'4'\
                    +',{:1.0f}'.format(self.ijk[0])\
                    +',{:1.0f}'.format(self.ijk[1])\
                    +',{:1.0f}'.format(self.ijk[2])\
                    +',{:1.0f}'.format(self.ijk[3])\
                    +var_str+'\n') 
        elif self.nD == 5:
            self.f.write('*DIM,'+self.name+','+array_type[:3]+'5'\
                    +',{:1.0f}'.format(self.ijk[0])\
                    +',{:1.0f}'.format(self.ijk[1])\
                    +',{:1.0f}'.format(self.ijk[2])\
                    +',{:1.0f}'.format(self.ijk[3])\
                    +',{:1.0f}'.format(self.ijk[4])\
                    +var_str+'\n')
            
        
    def locate(self, i, cont=1, place=[0,0,0,0,0]):
        loc = '('
        for j in range(self.nD):
            if i == j:
                loc = loc+str(cont)
            else:
                loc = loc+str(place[j])
            if j+1 < self.nD:
                loc = loc+','
        loc = loc+')'
        return loc
        
    def format_row(self, snip):
        row_str = ''        
        for num in snip:
            row_str = row_str+',{:3.7e}'.format(num)
        row_str = row_str[1:]
        return row_str
        
    def list_row(self, line):
        nTen = int(np.floor(len(line)/10))
        count,row_str = 0,[]
        for i in range(nTen):
            row_str.append(self.format_row(line[count:count+10]))
            count += 10
        if len(line) > count:
            row_str.append(self.format_row(line[count:]))
        return row_str
        
    def write_line(self, dimension, line, place=[0,0,0,0,0], TAXIS=0):
        row_str = self.list_row(line)
        cont = 1
        for row in row_str:
            loc = self.locate(dimension, cont=cont, place=place)
            if TAXIS == 1:
                self.f.write('*TAXIS,'+self.name+loc+','+str(dimension+1)+','+row+'\n')
            else:
                self.f.write(self.name+loc+'='+row+'\n')
            cont += 10
      
    def write_index(self):
        #self.f.write('! index\n')
        for i in range(len(self.index)):
            self.write_line(i, self.index[i], place=[1,1,1,1,1], TAXIS=1)
                
    def write_data(self):
        #self.f.write('! data\n')
        for m in range(int(self.ijk[4])):
            for l in range(int(self.ijk[3])):
                for k in range(int(self.ijk[2])):
                    for j in range(int(self.ijk[1])):
                        if self.nD == 1:
                            line = self.data
                        elif self.nD == 2:
                            line = self.data[:,j]
                        elif self.nD == 3:
                            line = self.data[:,j,k]
                        elif self.nD == 4: 
                            line = self.data[:,j,k,l]
                        elif self.nD == 5: 
                            line = self.data[:,j,k,l,m]
                        place = np.array([0,j,k,l,m])+1
                        self.write_line(0, line, place=place)

from subprocess import Popen, PIPE
from shlex import split
import os.path

def class_dir(name):
    root = name.__path__[0]
    return root

def trim_dir(check_dir):
    nlevel,dir_found = 3,False
    for i in range(nlevel):
        if os.path.isdir(check_dir):
            dir_found = True
            break
        else:
            if '../' in check_dir:
                check_dir = check_dir.replace('../','',1)
    if not dir_found:
        errtxt = check_dir+' not found\n'
        raise ValueError(errtxt)
    return check_dir

def qsub(script,jobname='analysis',t=1,freiahost=1,
         verbose=True,Code='Nova'):
    p = Popen(split('ssh -T freia{:03.0f}.hpc.l'.format(freiahost)),
              stdin=PIPE,stdout=PIPE,stderr=PIPE)
    p.stdin.write('cd ~/Code/Nova/nova\n'.encode()) 
    
             
    if '.py' not in script:
        script += '.py'
    #wd = 
    py3 = '~/Code/anaconda3/bin/python3'
    flags =  '-V -N {} -j y -wd -S {}'.format(jobname,wd,py3)
    if t > 1:
        flags += ' -t 1:{:1.0f}'.format(t) 
    flags += ' '
    qsub = 'qsub '+flags+script+'\n'
       
    #
    p.stdin.write(qsub.encode())  # submit job  
    p.stdin.flush()
    stdout,stderr = p.communicate()
    if verbose:
        print(stdout.decode(),stderr.decode())

class PATH(object):  # file paths

    def __init__(self,jobname,overwrite=True):
        import datetime
        import os

        self.path = {}
        self.jobname = jobname

        self.date_str = datetime.date.today().strftime('%Y_%m_%d')  # today
        
        # get root dir
        wd = os.getcwd()
        os.chdir('../')
        root = os.getcwd()
        os.chdir(wd)
        
        self.data = self.make_dir(root+'/Data')  # data dir
        if overwrite:
            self.folder = self.rep_dir(self.data+'/'+self.date_str+'.'+self.jobname)
        else:
            self.folder = self.make_dir(self.data+'/'+self.date_str+'.'+self.jobname)
        self.logfile = self.folder+'/run_log.txt'
      
    def new(self, task):
        self.task = task
        if task is 0:
            self.job = self.folder
        else:
            self.job = self.make_dir(self.data+'/'+self.date_str+
                                     '.'+self.jobname)
            self.job = self.rep_dir(self.data+'/'+self.date_str+
                                    '.'+self.jobname+'/task.'+str(task))
        self.screenID = self.jobname+'_task-'+str(task)
        
        # copy config files

        
    def make_dir(self,d):  # check / make
        import os
        if not os.path.exists(d):
            os.makedirs(d)
        return d
            
    def rep_dir(self,d):  # check / replace
        import os
        import shutil as sh
        if os.path.exists(d):
            sh.rmtree(d)
        os.makedirs(d)
        return d

    def go(self):
        import os
        self.home = os.getcwd()
        os.chdir(self.job)

    def back(self):
        import os
        os.chdir(self.home)
        
class SET_PATH(object):  # file paths

    def __init__(self,jobname,**kw):
        import os
        import datetime
        self.os = os
        self.file = {}
        self.jobname = jobname

        if 'date' in kw.keys():
            self.date_str = kw['date']
        else:
            self.date_str = datetime.date.today().strftime('%Y_%m_%d')  # today
        
        # get root dir
        self.home = self.os.getcwd()
        os.chdir('../')
        self.root = os.getcwd()
        os.chdir(self.home)
        
        self.data = self.check_dir(self.root+'/Data')  # data
        self.folder = self.check_dir(self.data+'/'+self.date_str+'.'+jobname)  # job

 
    def check_dir(self,d):  # check / replace
        if not self.os.path.exists(d):
            print(d)
            print('dir not found')
        return d
        
    def go(self,task):
        self.job = self.check_dir(self.data+'/'+self.date_str+'.'+self.jobname+
                                 '/task.'+str(task))  # job
        if 'Rfolder' in self.__dict__.keys():
            self.Rjob = self.make_dir(self.Rfolder+'/task.'+str(task))
        self.os.chdir(self.job)
        
    def back(self):
        self.os.chdir(self.home)
        
    def goto(self,there):
        self.home = self.os.getcwd()
        self.os.chdir(there)
        
    def make_dir(self, d):  # check / make
        import os
        if not os.path.exists(d):
            os.makedirs(d)
        return d
        
    def result(self,task=0,postfix=''):
        import shutil as sh
        self.Rdata = self.make_dir(self.root+'/Results')  # data
        self.Rfolder = self.make_dir(self.Rdata+'/'+
                                     self.date_str+'.'+self.jobname+postfix)    
        # initalise folder + copy inputs

        
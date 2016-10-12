
class PATH(object):  # file paths

    def __init__(self,jobID,scriptID,working_fluid,overwrite=True,**kw):
        import datetime
        import os

        self.path = {}
        self.scriptID = scriptID
        self.jobID = jobID
        self.fluid = working_fluid
        self.date_str = datetime.date.today().strftime('%Y_%m_%d')  # today
        
        # get root dir
        wd = os.getcwd()
        os.chdir('../')
        root = os.getcwd()
        os.chdir(wd)
        
        self.data = self.make_dir(root+'/Data')  # data dir
        if overwrite:
            self.folder = self.rep_dir(self.data+'/'+self.date_str+'.'+self.jobID)
        else:
            self.folder = self.make_dir(self.data+'/'+self.date_str+'.'+self.jobID)
        self.logfile = self.folder+'/run_log.txt'
      
    def new(self, task):
        self.task = task
        import shutil as sh
        
        if task is 0:
            self.job = self.folder
        else:
            self.job = self.make_dir(self.data+'/'+self.date_str+
                                     '.'+self.jobID)
            self.job = self.rep_dir(self.data+'/'+self.date_str+
                                    '.'+self.jobID+'/task.'+str(task))
        self.screenID = self.jobID+'_task-'+str(task)
        
        if task >= 0:
            # copy config files
            for script in self.scriptID:  
                sh.copyfile('../Config/'+script+'.py', self.job+'/'+script+'.py')
            sh.copyfile(self.job+'/'+self.scriptID[0]+'.py', self.job+'/'+'config.py')
            
            #if task > 0: 
            #    sh.copyfile(self.folder+'/input.pk', self.job+'/input.pk')
    
            # submision scripts
            self.mgen = self.jobID+'.mgen'
            self.mcor = self.jobID+'.mcor'
            
            # file paths
            self.file = {}
            extension = ['GEN_OUT','GEN_DIA', 'COR_OUT','COR_DIA','RST','PTF','Data']
            for ex in extension:
                self.file[ex] = ex+'.vi'
            
            # copy material data, 'n2' 'he', 'h2o'
            sh.copyfile('../Materials/'+'tpf'+self.fluid,
                        self.job+'/'+'tpf'+self.fluid)
        
    def make_dir(self, d):  # check / make
        import os
        if not os.path.exists(d):
            os.makedirs(d)
        return d
            
    def rep_dir(self, d):  # check / replace
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

    def __init__(self, jobID, **kw):
        import os
        import datetime
        self.os = os
        self.file = {}
        self.jobID = jobID

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
        self.folder = self.check_dir(self.data+'/'+self.date_str+'.'+jobID)  # job

        self.file['PTF'] = 'PTF.vi'
        self.file['Data'] = 'Data.vi'
        
    def check_dir(self, d):  # check / replace
        if not self.os.path.exists(d):
            print(d)
            print('dir not found')
        return d
        
    def go(self,task):
        self.job = self.check_dir(self.data+'/'+self.date_str+'.'+self.jobID+
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
                                     self.date_str+'.'+self.jobID+postfix)    
        if task == 0:  # initalise folder + copy inputs
            sh.copyfile(self.folder+'/run_log.txt',self.Rfolder+'/run_log.txt')
        else:
            self.Rjob = self.make_dir(self.Rfolder+'/task.'+str(task))
        
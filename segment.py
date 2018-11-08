
import cv2
import numpy as np
import queue
# import rotation

class Segmentation:
    threshold=60
    addht=0
    addwd=0
    def __init__(self):
        self.cc=0
        # self.aa=0
        '''if fn is not None:
            self.xmin,self.ymin=7000,7000
            self.xmax,self.ymax=0,0
            self.pixs=np.array([],dtype=np.uint8)

            self.filename=fn
            self.filename2=fn
        
            self.ax,self.ay=0,0
            self.img=cv2.imread(self.filename,0)
            self.imgOriginal=cv2.imread(self.filename2)
            self.sx,self.sy=self.img.shape'''
    
    def setData(self, fn):
        if fn is not None:
            self.xmin,self.ymin=7000,7000
            self.xmax,self.ymax=0,0
            self.pixs=np.array([],dtype=np.uint32)
            self.linepixs=np.array([],dtype='int32')
            # self.pathchars="//home//qais//OCRProject//chars//"
            self.filename=fn
        
            self.ax,self.ay=0,0
            self.imgOriginal=cv2.imread(self.filename)
            self.img=cv2.imread(self.filename,0)
            # self.img=cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)

            
            if self.img is None:
                print("please pass a valid filename with extension ")
                exit()
            self.clone=self.img.copy()
            self.sx,self.sy=self.img.shape
        
    def resize(self):
        
        if self.sx>1500:
             
            self.img=Segmentation.image_resize(self.img,height=1500)
            self.imgOriginal=Segmentation.image_resize(self.imgOriginal,height=1500)
            self.clone=self.img.copy()

            self.sx,self.sy=self.img.shape
         
        
        if self.sy>1500:
            # print("hope this never gets executed")
            self.img=Segmentation.image_resize(self.img,width=1500)
            self.imgOriginal=Segmentation.image_resize(self.imgOriginal,width=1500)
            self.clone=self.img.copy()

            self.sx,self.sy=self.img.shape
         

       
        '''
        if(self.sx > 3000):
            self.img=Segmentation.image_resize(self.img,height=int(self.sx*.40))
            self.imgOriginal=Segmentation.image_resize(self.imgOriginal,height=int(self.sx*.40))
            self.sx,self.sy=self.img.shape

            if(self.sy>1000):
                self.Segmentation.image_resize(self.img,width=1000)
                self.imgOriginal=Segmentation.image_resize(self.imgOriginal,width=1000)
                sx,sy=img.shape
        if(self.sx>2000):
            self.img=Segmentation.image_resize(self.img,height=int(self.sx*.50))
            self.imgOriginal=Segmentation.image_resize(self.imgOriginal,height=int(self.sx*.50))
            self.sx,self.sy=self.img.shape
            if(sy>1000):
                img=imageResize.image_resize(img,width=1000)
                imgOriginal=imageResize.image_resize(imgOriginal,width=1000)
                sx,sy=img.shape

	'''
    def modifyImage(self,img,vb,hb):
        
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,\
                cv2.THRESH_BINARY,15,7)
        r,img=cv2.threshold(img,75,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.sx,self.sy= img.shape
        
        r,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((vb,hb),np.uint8)
        
        img = cv2.dilate(img,kernel,iterations = 1)
        # self.img=cv2.Canny(self.img,100,200)
        
          
        r,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        img=cv2.Laplacian(img,cv2.CV_64F)
        # self.img=cv2.Canny(self.img,100,200)
        r,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        cv2.imwrite("blobl.jpg",img)

        return img

    def modifyLines(self,img,vb,hb):
        
        # self.img = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,15,7)
        # r,self.img=cv2.threshold(self.img,75,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # self.sx,self.sy= self.img.shape
        
        r,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY_INV )

        kernel = np.ones((vb,hb),np.uint8)
        
        img = cv2.dilate(img,kernel,iterations = 1)
        # img = cv2.erode(img,kernel,iterations = 2)
        # self.img=cv2.Canny(self.img,100,200)
        
          
        r,img = cv2.threshold(img,20,255,cv2.THRESH_BINARY_INV)
        # self.img=cv2.Laplacian(self.img,cv2.CV_64F)
        # self.img=cv2.Canny(self.img,100,200)
        # r,self.img = cv2.threshold(self.img,127,255,cv2.THRESH_BINARY_INV)
        cv2.imwrite("blob.jpg",img)

        return img
    
    @classmethod
    def image_resize(cls, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

            # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    
    

    def segments(self,img,x,y):
        if self.mode==False:
            # segmenting lines
            self.xmin,self.ymin=img.shape
            self.xmax,self.ymax=0,0
            sx=self.sx
            sy=self.sy
        else:
            # segmenting chars
            self.xmin=self.linedim[0]
            self.xmax=self.linedim[2]
            self.ymin=self.linedim[3]
            self.ymax=self.linedim[1]

            sx=self.linedim[2]+1
            sy=self.sy

    
        q=queue.Queue()
    
        if(img[x][y]==0):
            img[x][y]=50
        
            q.put([x,y])
        
            if(self.xmin>x ):
                self.xmin=x
            if(self.ymin>y):
                self.ymin=y
        
            while( 1==1 ):
                #print("1==1")
            
                while( y+1<sy and img[x][y+1]==0):
                    y=y+1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
            
                if(y>self.ymax):
                    self.ymax=y
                
                while( y+1<sy and x+1 < sx and img[x+1][y+1]==0):
                    y=y+1
                    x=x+1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
            
                if(y>self.ymax):
                    self.ymax=y
                if(x>self.xmax):
                    self.xmax=x
            
                while(y-1>=0 and img[x][y-1]==0):
                    y=y-1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
                    
                while(y-1>=0 and x-1>0 and img[x-1][y-1]==0):
                    y=y-1
                    x=x-1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
                    
                if(self.ymin>y):
                    self.ymin=y
                if(self.xmin>x):
                    self.xmin=x
            
                while(x-1>=0 and img[x-1][y]==0):
                    x=x-1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
                if(self.xmin>x):
                    self.xmin=x
            
                while(x-1>=0 and y+1<sy and img[x-1][y+1]==0):
                    x=x-1
                    y=y+1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
            
                if(self.xmin>x):
                    self.xmin=x
                if(y>self.ymax):
                    self.ymax=y
            
                while(x+1<sx and img[x+1][y]==0):
                    x=x+1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
                    
                if(x>self.xmax):
                    self.xmax=x
                
                while(x+1<sx  and y-1 > 0 and img[x+1][y-1]==0):
                    x=x+1
                    y=y-1
                    if(img[x][y]==0):
                        img[x][y]=50
                    
                        q.put([x,y])
                if(self.ymin>y):
                    self.ymin=y
                
                if (q.empty()):
                    break
                else:
                    x,y=q.get()

            #     cv2.imshow('result',img)
            #     cv2.waitKey(10)
            
            # cv2.destroyAllWindows()

                # print(self.ymin,self.ymax)
            if self.mode==True:
                self.xmin=self.linedim[0]
                self.xmax=self.linedim[2]

            self.pixs=np.append(self.pixs,[self.xmin,self.ymin,self.xmax,self.ymax])  
            # print(self.ymin,self.ymax)
        return img
    
        # if (self.flag==1):
        
        #     if(self.linepixs.size>0):
            
        #         self.linepixs=self.linepixs.reshape((int(self.linepixs.size/4),4))
        #         self.linepixs = self.linepixs[self.linepixs[:,1].argsort()]
        #         self.pixs=np.append(self.pixs,self.linepixs)
        #         self.linepixs=np.array([],dtype='int32')
            

                
            
    def fixPix(self):
    
    
        self.pixs=self.pixs.reshape((int(self.pixs.size/4),4))
        
        return

    def findAvgSize(self):
    
        # self.ax=np.mean(self.pixs[:,2]-self.pixs[:,0])
        # if self.ax+5>Segmentation.threshold :
        #     Segmentation.threshold=self.ax+5

        self.ay=np.mean(self.pixs[:,3]-self.pixs[:,1])
    
    def findSpaces(self):
        
        i=0
        while(i < (self.pixs.size/4)-1 and (self.pixs.size/4) > 1):
            if ( abs(self.pixs[i][3]-self.pixs[i+1][1]) > self.ay*0.5 ):
                #pixs=np.insert(pixs,i+1,[pixs[i][0],pixs[i][3]+1,pixs[i][2],5],axis=0)
                # print("space")
                self.pixs=np.insert(self.pixs,i+1,[-1,-1,-1,-1],axis=0)
                i=i+2
        
            i=i+1

        return
    
    def selectSegments(self,pixs):
    
        i=0
    
        while(i<int(pixs.size/4)):
        
            if  ( pixs[i][2]-pixs[i][0] > Segmentation.threshold ):
                pixs= np.delete(pixs, i,  axis=0)
                i=i-1
            elif ( pixs[i][2]-pixs[i][0] < 10 and pixs[i][3]-pixs[i][1] < 5 and self.mode == False):
                pixs=np.delete(pixs,i,axis=0)
                i=i-1
            i=i+1
   
        # removing concentric segments if any before

        j=0
        i=0

        while(i<pixs.size/4):
            j=0
            while(j<i):
                if(pixs[j][0] <= pixs[i][0] and pixs[j][2] >= pixs[i][2] and pixs[j][1]<=pixs[i][1] and pixs[j][3]>=pixs[i][3]):
                    # print("deleting : ",pixs[i])
                    pixs=np.delete(pixs,i,axis=0)

                    i-=1
                        #s-=1
                        #break
                j+=1
            
            i+=1

        
        # removing concentric segments if any after
        
        i,j=0,0

        while(i<pixs.size/4):
            j=i
            while(j<pixs.size/4):
                if(pixs[i][0]<pixs[j][0] and pixs[i][2]>pixs[j][2] ):
                    if(pixs[i][1]<pixs[j][1] and pixs[i][3]>pixs[j][3]):
                        pixs=np.delete(pixs,j,axis=0)
                        j-=1
                        #s-=1
                        #break
                j+=1
            
            i+=1

        
        return pixs


    def duplicateChars(self):
        #lkxl
        i=0
        while(i<self.chars.size/4):
            j=i+1
            while(j<i+2 and j < self.chars.size/4 ):
                # print(i,j)
               
                # remove overlapping
                #         j-xmin  < i-xmax +1   and                   j-xamx > i-xmax
                if ( self.chars[j][1] < self.chars[i][3] - 3 and self.chars[j][3] > self.chars[i][3] +1 and j > 0):
                    self.chars[i][3] = self.chars[j][3]
                    if self.chars[j][1]< self.chars[i][1] :
                        self.chars[i][1] = self.chars[j][1]
                    self.chars=np.delete(self.chars,j,axis=0)

                    j-=1

                j+=1
            
            i+=1

        #kjkj
        

        i=0
        while(i<self.chars.size/4):
            j=i+1
            while(j<i+2 and j < self.chars.size/4 ):

                 # remove concentric
                if( self.chars[j][1]>=self.chars[i][1] and self.chars[j][3]<=self.chars[i][3]):
                    # print("deleting : ",self.chars[i])
                    self.chars=np.delete(self.chars,j,axis=0)

                    j-=1
 
                j+=1
            
            i+=1




    def prepareReturn(self):

        i=0
        self.sx,self.sy=self.img.shape
        s=self.pixs.size/4
        while (i<s):
            

	
       
            '''
            #converting pixs into x,y,h,w only
            ht=self.pixs[i][2]-self.pixs[i][0]
            wd=self.pixs[i][3]-self.pixs[i][1]
            self.pixs[i][2]=ht
            self.pixs[i][3]=wd
            '''

            #adding addht to height and addwd to width and converting to x,y,h,w
            ht=self.pixs[i][2]-self.pixs[i][0]
            wd=self.pixs[i][3]-self.pixs[i][1]
            self.pixs[i][0]=self.pixs[i][0]-Segmentation.addht
            self.pixs[i][1]=self.pixs[i][1]-Segmentation.addwd
            self.pixs[i][2]=ht+Segmentation.addht+Segmentation.addht
            self.pixs[i][3]=wd+Segmentation.addwd+Segmentation.addwd

            
    	    #code to fix pixs as actual x,y,h,w
            a,b,c,d=self.pixs[i]
            self.pixs[i][0]=b   #x
            self.pixs[i][1]=a   #y
            self.pixs[i][2]=d   #w
            self.pixs[i][3]=c   #h
            

            i+=1

    def makerects(self):
        
        for x in self.wordpixs:
            
            #cv2.rectangle(self.imgOriginal,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),(0,0,255),1)
            cv2.rectangle(self.imgOriginal,(x[1],x[0]),(x[3],x[2]),(0,0,255),1)
            # cv2.imshow("result",self.imgOriginal)
            # cv2.waitKey(50)

        cv2.imwrite("blobout.jpg",self.imgOriginal)
        # self.aa+=1
        # print("de nazar")
        cv2.destroyAllWindows()
    
    def croplines(self):
        a=0
        for x in self.linepixs:
            i=self.clone[x[0]-2:x[2],x[1]:x[3]]
            # _,i=cv2.threshold(i,127,255,cv2.THRESH_BINARY)
            cv2.imwrite(str(a)+".jpg",i)
            a=a+1
        #     self.lines=np.append(self.lines,[i])

        # self.lines=self.lines.reshape((-1,4))



    def cropChars(self):
        # i=self.cc
        for x in self.chars:
            #if( pixs[i][2]-pixs[i][0] > ax/1.5 and pixs[i][3]-pixs[i][1]< ay*5):
            c=str(self.cc)

            if(x[1]==-1):
                temp=np.zeros((30,30))+255
                cv2.imwrite(self.pathchars+c+".jpg",temp)
                
            if(x[1]!=-1):
                self.crop(c,x[1]-1,x[3]+1,x[0]-1,x[2]+1)
            self.cc+=1

    def crop(self,c,y1,y2,x1,x2):

        
        i=self.imgOriginal[x1:x2,y1:y2]
        
        i=cv2.cvtColor( i, cv2.COLOR_RGB2GRAY )

        i = cv2.threshold(i,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cv2.imwrite(self.pathchars+c+".jpg",i)
    
    def doSegmentation(self,filename=None,out=None,size_thresh=None):
        # self.cc=temp
        if size_thresh is None:
            Segmentation.threshold=55
        elif size_thresh is  not None:
             Segmentation.threshold=size_thresh
        #self.__init__(self,filename)
        if filename is None or filename=='' :
            print("please pass filename with extension ")
            print(filename)
            exit()
        self.mode=False
        
        self.setData(filename)
        self.resize()
        self.img=self.modifyImage(self.img,3,5)
        i=self.sx
        j=self.sy

        #horizontal checking
        for x in range(0,i-2,1):
            for y in range(0,j-2,1):
                self.img=self.segments(self.img,x,y)
            
        self.fixPix()
        # self.linepixs=self.pixs
        # print(self.pixs.shape)
        self.wordpixs = self.selectSegments(self.pixs)
        # self.wordpixs=self.pixs
        # print(self.wordpixs.shape)
        # self.makerects()
        target='text'

        import pytesseract

        for x in self.wordpixs:
            i=self.clone[x[0]-2:x[2],x[1]:x[3]]
            # _,i=cv2.threshold(i,127,255,cv2.THRESH_BINARY)
            word = pytesseract.image_to_string(i)
            # print(word)
            if word.lower().strip() == target :
                # print(word.lower().strip())
                cv2.rectangle(self.imgOriginal,(x[1]-1,x[0]-1),(x[3]+1,x[2]+1),(0,0,255),2)

        cv2.imwrite('out1.jpg',self.imgOriginal)

if __name__=="__main__":
    obj=Segmentation()
    obj.doSegmentation(filename="img.jpg",out='out.jpg',size_thresh=150,word_dict={'good':{'green':['text','style']},'bad':{'yellow':['each']}})

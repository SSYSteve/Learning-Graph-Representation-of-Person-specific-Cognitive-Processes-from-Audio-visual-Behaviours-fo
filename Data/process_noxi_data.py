import sys
import os
import subprocess
import pickle
import csv
import utils
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import argparse
parser = argparse.ArgumentParser("data-process")
parser.add_argument('--vidio_dir', type=str)
parser.add_argument('--openface_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()


class VideoFrame():
    def __init__(self,FFrame,AFrame,FacePath):
        self.FL = FFrame
        self.FacePath = FacePath
        self.MFCC = AFrame

class AudioProcessor():
    def __init__(self,videoDirPath):
        self.videoDirPath = videoDirPath
        self.ExpertDir,self.NoviceDir = self.sortByIDListIDFiles(self.videoDirPath,'mp4')
        self.MFCC = None


    def sortByIDListIDFiles(self,rootPath,suffix=None):
        assert suffix,"suffix can't be None"
        expert = []
        novice = []
        listIDsPath = os.listdir(rootPath)
        listIDsPath.sort()
        for ID in listIDsPath:
            if os.path.isdir(os.path.join(rootPath,ID)):
                for file in os.listdir(os.path.join(rootPath,ID)):
                    if file == 'Expert_video.'+suffix:
                        expert.append(os.path.join(rootPath,ID,file))
                    elif file == 'Novice_video.'+suffix:
                        novice.append(os.path.join(rootPath,ID,file))
                    else:
                        pass
        assert len(expert)==len(novice),"num of experts should be same as novices"
        return expert,novice

    def extractSaveRawAudio(self,saveDir):
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        for i in range(len(self.ExpertDir)):
            saveIDDir = os.path.join(saveDir,self.ExpertDir[i].split('/')[-2])
            if not os.path.exists(saveIDDir):
                os.mkdir(saveIDDir)
            cmdLine = 'ffmpeg -i %s -vn -ac 1 -ar 48000 %s'%(self.ExpertDir[i],os.path.join(saveIDDir,self.ExpertDir[i].split('/')[-1]).replace('mp4','wav'))
            subprocess.run(cmdLine,shell=True)
            cmdLine = 'ffmpeg -i %s -vn -ac 1 -ar 48000 %s'%(self.NoviceDir[i],os.path.join(saveIDDir,self.NoviceDir[i].split('/')[-1]).replace('mp4','wav'))
            subprocess.run(cmdLine,shell=True)
            print(i,end=' ')

    def extractSaveMFCC(self,AudioDir):
        def getMFCC(sourcePath):
            (rate,sig) = wav.read(sourcePath)
            result = mfcc(sig,rate,winlen=0.04,winstep=0.04,numcep=128,nfilt=256,nfft=int(rate*0.04))
            return result
        ExpertAudioDir,NoviceAudioDir = self.sortByIDListIDFiles(AudioDir,'wav')
        MFCCPool = [ [None,None] for i in range(len(ExpertAudioDir))]
        for i in range(len(ExpertAudioDir)):
            MFCCPool[i][0] = getMFCC(ExpertAudioDir[i])
            MFCCPool[i][1] = getMFCC(NoviceAudioDir[i])
            print(i,end = ' ')
        self.MFCC = MFCCPool

    def getMFCC(self,rawAudioSavePath,MFCCSavePath):
        print("extract audio from %s and save .wav at %s"%(self.videoDirPath,rawAudioSavePath))
        self.extractSaveRawAudio(rawAudioSavePath)
        self.extractSaveMFCC(MFCCSavePath)

class VideoProcessor():
    def __init__(self,openfaceDirPath,audios=None):
        self.openfaceDirPath = openfaceDirPath
        self.faceLms = None
        self.faceImgPaths = None
        self.audios = audios
        self.videoFrames = None
        assert audios,"audio can't be None"
        self.readOpenfaceCSV()
        self.listFaceImgPaths()


    def sortByIDListIDFiles(self,rootPath,suffix=None):
        assert suffix,"suffix can't be None"
        expert = []
        novice = []
        listIDsPath = os.listdir(rootPath)
        listIDsPath.sort()
        for ID in listIDsPath:
            if os.path.isdir(os.path.join(rootPath,ID)):
                for file in os.listdir(os.path.join(rootPath,ID)):
                    if file == 'Expert_video.'+suffix:
                        expert.append(os.path.join(rootPath,ID,file))
                    elif file == 'Novice_video.'+suffix:
                        novice.append(os.path.join(rootPath,ID,file))
                    else:
                        pass
        assert len(expert)==len(novice),"num of experts should be same as novices"
        return expert,novice

    def readOpenfaceCSV(self):
        def SplitCsv(csvList):
            Flmark=[]
            for i in range(len(csvList)):
                print(i,end=' ')
                csvFile = open(csvList[i])
                reader = csv.reader(csvFile)
                frame = []
                for item in reader:
                    #                 if reader.line_num == 1 or int(item[4])==0:
                    if reader.line_num == 1:
                        continue
                    frame.append([[float(item[j]),float(item[j+68])] for j in range(5,5+68)])
                Flmark.append(np.array(frame))
                csvFile.close()
            return Flmark
        expert,novice = self.sortByIDListIDFiles(self.openfaceDirPath,'csv')
        FlmarkExpert = SplitCsv(expert)
        FlmarkNovice = SplitCsv(novice)
        self.faceLms = [[seq1,seq2] for seq1,seq2 in zip(FlmarkExpert,FlmarkNovice)]

    def listFaceImgPaths(self):
        expert = []
        novice = []
        listDirDirPath = os.listdir(self.openfaceDirPath)
        listDirDirPath.sort()
        for talk in listDirDirPath:
            if os.path.isdir(os.path.join(self.openfaceDirPath,talk)):
                for file in os.listdir(os.path.join(self.openfaceDirPath,talk)):
                    if file == 'Expert_video_aligned':
                        expert.append(os.path.join(talk,file))
                    elif file == 'Novice_video_aligned':
                        novice.append(os.path.join(talk,file))
                    else:
                        pass
        assert len(expert)==len(novice)
        self.faceImgPaths = [[path1,path2] for path1,path2 in zip(expert,novice)]

    def package_video(self):
        self.readOpenfaceCSV()
        self.listFaceImgPaths()
        VideoPool = [ [None,None] for i in range(len(self.faceLms))]
        print("------package video into video Frames witch contians : FaceLandMarks, FaceImagePath, MFCC")
        for i in range(len(VideoPool)):
            print(i)
            #先处理一下Expert
            print('delta length of Novice video and audio: ',self.faceLms[i][0].shape[0]-len(self.audios[i][0]))
            videoLen = min(self.faceLms[i][0].shape[0],len(self.audios[i][0]))
            VideoPool[i][0] = [VideoFrame(self.faceLms[i][0][index],self.audios[i][0][index],os.path.join(self.faceImgPaths[i][0],'frame_det_00_'+str(index+1).zfill(6)+'.bmp')) for index in range(videoLen)]
            #再处理一下Novice
            print('delta length of Expert video and audio: ',self.faceLms[i][1].shape[0]-len(self.audios[i][1]))
            videoLen = min(self.faceLms[i][1].shape[0],len(self.audios[i][1]))
            VideoPool[i][1] = [VideoFrame(self.faceLms[i][1][index],self.audios[i][1][index],os.path.join(self.faceImgPaths[i][1],'frame_det_00_'+str(index+1).zfill(6)+'.bmp')) for index in range(videoLen)]
        self.videoFrames = VideoPool

        print("------make sure Novice and Expert Frames have same length")
        for i in range(len(self.videoFrames)):
            print(i,' ','delta length of Novice and Expert: ',len(self.videoFrames[i][0])-len(self.videoFrames[i][1]))
            videoLen = min(len(self.videoFrames[i][0]),len(self.videoFrames[i][1]))
            self.videoFrames[i][0] = self.videoFrames[i][0][:videoLen]
            self.videoFrames[i][1] = self.videoFrames[i][1][:videoLen]

    def remove_bad_frames(self):
        def findBadIndex(expert,novice):
            badFrameIndex = []
            for i,(csv1,csv2) in enumerate(zip(expert,novice)):
                badFrameIndexTemp = []
                print(i,end=' ')
                csvFile = open(csv1)
                reader = csv.reader(csvFile)
                for item in reader:
                    if reader.line_num == 1:
                        continue
                    if int(item[4])==0:
                        badFrameIndexTemp.append(reader.line_num-2)
                csvFile.close()
                csvFile = open(csv2)
                reader = csv.reader(csvFile)
                for item in reader:
                    if reader.line_num == 1:
                        continue
                    if int(item[4])==0:
                        badFrameIndexTemp.append(reader.line_num-2)
                csvFile.close()
                badFrameIndex.append(badFrameIndexTemp)
            return badFrameIndex
        print("------removing failed detected frames")
        expert,novice = self.sortByIDListIDFiles(self.openfaceDirPath,'csv')
        badFrameIndex = findBadIndex(expert,novice)
        for i in range(len(expert)):
            tempEx = []
            tempNo = []
            for index in range(len(self.videoFrames[i][0])):
                if index in badFrameIndex[i]:
                    print(index,end = '- ')
                    continue
                tempEx.append(self.videoFrames[i][0][index])
                tempNo.append(self.videoFrames[i][1][index])
            self.videoFrames[i][0] = tempEx
            self.videoFrames[i][1] = tempNo

    def FaceAlign(self):
        print('------align face landmark')
        f1 = open('./trainMeanFace','rb')
        meanFrame = pickle.load(f1)
        meanFrame = np.array(meanFrame)
        meanFrame = meanFrame.reshape(-1,136).mean(0).reshape(68,2)
        f1.close()
        t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
        meanShape = meanFrame[t_shape_idx,:]
        for i in range(len(self.videoFrames)):
            print(i,end=' ')
            for index,frame in enumerate(self.videoFrames[i][0]):
                tempShape = frame.FL[t_shape_idx,:]
                T, distance, itr = utils.icp(tempShape, meanShape)
                rot_mat = T[:2, :2]
                trans_mat = T[:2, 2:3]
                self.videoFrames[i][0][index].FL = np.dot(rot_mat, frame.FL.T).T + trans_mat.T
            for index,frame in enumerate(self.videoFrames[i][1]):
                tempShape = frame.FL[t_shape_idx,:]
                T, distance, itr = utils.icp(tempShape, meanShape)
                rot_mat = T[:2, :2]
                trans_mat = T[:2, 2:3]
                self.videoFrames[i][1][index].FL = np.dot(rot_mat, frame.FL.T).T + trans_mat.T
        print('------normalize face landmark and MFCC')
        FLPool = [ [None,None] for i in range(len(self.videoFrames))]
        for i in range(len(self.videoFrames)):
            temp = [frame.FL for frame in self.videoFrames[i][0]]
            temp = np.array(temp)
            FLPool[i][0] = temp
            temp = [frame.FL for frame in self.videoFrames[i][1]]
            temp = np.array(temp)
            FLPool[i][1] = temp

        for i in range(len(self.videoFrames)):
            print(i,end=' ')
            MAX = FLPool[i][0].max()
            MIN = FLPool[i][0].min()
            FLPool[i][0] = (FLPool[i][0]-MIN)/(MAX-MIN)
            print(MAX-MIN,end=' ')
            MAX = FLPool[i][1].max()
            MIN = FLPool[i][1].min()
            FLPool[i][1] = (FLPool[i][1]-MIN)/(MAX-MIN)
            print(MAX-MIN)


        for i in range(len(self.videoFrames)):
            for index in range(len(self.videoFrames[i][0])):
                self.videoFrames[i][0][index].FL = FLPool[i][0][index]
                self.videoFrames[i][1][index].FL = FLPool[i][1][index]

        MFPool = [ [None,None] for i in range(84)]
        for i in range(len(self.videoFrames)):
            temp = [frame.MFCC for frame in self.videoFrames[i][0]]
            temp = np.array(temp)
            MFPool[i][0] = temp
            temp = [frame.MFCC for frame in self.videoFrames[i][1]]
            temp = np.array(temp)
            MFPool[i][1] = temp

        for i in range(len(self.videoFrames)):
            print(i,end=' ')
            MAX = MFPool[i][0].max()
            MIN = MFPool[i][0].min()
            MFPool[i][0] = (MFPool[i][0]-MIN)/(MAX-MIN)
            MAX = MFPool[i][1].max()
            MIN = MFPool[i][1].min()
            MFPool[i][1] = (MFPool[i][1]-MIN)/(MAX-MIN)

        for i in range(len(self.videoFrames)):
            for index in range(len(self.videoFrames[i][0])):
                self.videoFrames[i][0][index].MFCC = MFPool[i][0][index]
                self.videoFrames[i][1][index].MFCC = MFPool[i][1][index]



    def _getSaveFrames(self,dir):
        print('------saving videoFrames to %s'%(dir))
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i in range(len(self.videoFrames)):
            currySaveDir = os.path.join(dir,str(i))
            f1 = open(currySaveDir,'wb')
            pickle.dump(self.videoFrames[i],f1)
            f1.close()

    def getSaveFrames(self,save_dir=None):
        assert save_dir,"save_dir can't be None"
        self.package_video()
        self.remove_bad_frames()
        self.FaceAlign()
        self._getSaveFrames(save_dir)


if __name__ == '__main__':
    ap = AudioProcessor(args.vidio_dir)
    ap.getMFCC(args.save_dir,args.save_dir)
    vp = VideoProcessor(args.openface_dir,ap.MFCC)
    vp.getSaveFrames(args.save_dir)
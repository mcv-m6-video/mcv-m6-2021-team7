import cv2


class VideoModel:
    def __init__(self,path,color_space):
        self.path = path
        self.vCapture = cv2.VideoCapture(path)
        self.width = int(self.vCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.length = int(self.vCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.color_space = color_space
        self.vidLen = int(self.vCapture.get(cv2.CAP_PROP_FRAME_COUNT))

    def retVidLen(self):
        return self.vidLen




import matplotlib.pyplot as plt

class PlotCreator:
    def __init__(self):
        super(PlotCreator, self).__init__()
    def plotCurve(self,datax,datay,labelx,labely,name):
        plt.plot(datax, datay)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.savefig(name+'.png')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Cursor, Button

import Layers

class GUI:
    #============ define data
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'r', 'g', 'b', 'm', 'c', 'y', 'r', 'g', 'b', 'm', 'c', 'y', 'k']
    # It's inevitable to begine with 'b', as Radio's default active color is 'b'
    axcolor = 'w'

    model, loss, pred_dim = None, None, None
    classNames, currentClass = None, None
    x_list, y_list, currentLine = None, None, None

    #---------- create XY-meshgrid
    lim = 5.
    step = .25
    x = np.arange(-lim, lim + step, step).reshape(-1, 1)
    y = np.arange(-lim, lim + step, step).reshape(-1, 1)
    xy = None

    X, Y = np.meshgrid(x, y)
    XY = np.stack((X, Y), axis=-1)

    #========== interface to Trainer
    extTrain, extRemove, extLrUp, extLrDn, compileTitle  = None, None, None, None, None

    #=========== create and layout widgets
    fig, axes, classAxes, classRadios = None, None, None, None
    lines2d, cursors = None, None
 
    #---------- create button ugly controllers and buttons, and connect buttons
    axbTrain, axbRemove, axbLrUp, axbLrDn = None, None, None, None
    bTrain, bRemove, bLrUp, bLrDn = None, None, None, None

    mouseMoveBinding, mousePressBinding = None, None

    total_epochs = 0

    def funTrain(bEvent):
        if GUI.extTrain is not None:
            epochs = 10; minibatchsize = 10
            GUI.showTitle(GUI.compileTitle(), "Training...")
            trained = GUI.extTrain(epochs, minibatchsize)    # epochs and minibatchsize will be determined with the model size and data size.
            if trained:
                GUI.total_epochs += epochs
            GUI.showTitle(GUI.compileTitle(), "Done {:d} epochs".format(GUI.total_epochs))

    def funRemove(bEvent):
        if GUI.extRemove is not None:
            GUI.removePoints()
            GUI.extRemove()
            GUI.showTitle(GUI.compileTitle(), "Points removed.")
            GUI.total_epochs = 0
            # GUI.activateDataCreator()
            # refreshMainAxes(model, nContours=10)

    
    def funLrUp(bEvent):
        if GUI.extLrUp is not None:
            GUI.extLrUp()
            GUI.showTitle(GUI.compileTitle(), "Learning rate doubled.")

    def funLrDn(bEvent):
        if GUI.extLrDn is not None:
            GUI.extLrDn()
            GUI.showTitle(GUI.compileTitle(), "Learning rate halved.")

    def init(model, loss, pred_dim) -> None:
        GUI.model, GUI.loss, GUI.pred_dim = model, loss, pred_dim
        nColors = pred_dim if loss == "MCE" else (2 if loss == 'BCE' else 6)
        assert nColors > 0 and nColors < len(GUI.colors)
        GUI.colors = GUI.colors[:nColors]
        GUI.x_list = [[] for _ in GUI.colors]
        GUI.y_list = [[] for _ in GUI.colors]

        GUI.currentLine = 0
        GUI.classNames = ["Class " + str(n) for n in range(len(GUI.colors))]
        GUI.currentClass = GUI.classNames[GUI.currentLine]

        # GUI.xy = np.array([[(x, y) for x in GUI.x] for y in GUI.y], dtype=float)
        GUI.nAxes = pred_dim + 1 if GUI.loss == "MCE" else pred_dim
        assert GUI.nAxes > 0
        nRows, nCols = GUI.getRowsCols(GUI.nAxes)

        GUI.fig = plt.figure(figsize=(16, 9))
        GUI.axes = GUI.fig.subplots(nrows=nRows, ncols=nCols,  sharex=True, sharey=True)
        plt.subplots_adjust(left=0.2, wspace=0.02, hspace=0.02) #, wspace=0.05, hspace=0.001)
        GUI.axes = GUI.flatten2DList(GUI.axes)
        del GUI.axes[GUI.nAxes:]

        GUI.lines2d = [None] * GUI.nAxes
        GUI.cursors = [None] * GUI.nAxes

        for a in range(GUI.nAxes):
            GUI.axes[a].set_xlim(-GUI.lim, GUI.lim)
            GUI.axes[a].set_ylim(-GUI.lim, GUI.lim)
            GUI.lines2d[a] = [GUI.axes[a].plot(GUI.x_list[i], GUI.y_list[i], 'o', color=GUI.colors[i]) for i in range(len(GUI.colors)) ]
            GUI.cursors[a] = Cursor(GUI.axes[a], useblit=True, color='k', linewidth=0.5)

        #--------- Create radio buttons
        classLeftBottom = [0.025, 0.55]
        classWidthHeight = [0.15, 0.265]
        GUI.classAxes = GUI.fig.add_axes(classLeftBottom + classWidthHeight,
        facecolor=GUI.axcolor, frameon=False, label='Select a class (or Y)')
        GUI.classRadios = RadioButtons(GUI.classAxes, GUI.classNames)
        # Now, the default active color is 'b' and RadioButtons assumes the first item is clicked.
        GUI.classAxes.set_xlim(0,1)
        GUI.classAxes.set_ylim(0,1)
        for n in range(len(GUI.colors)):
            y = 1. - 1./(len(GUI.colors)+1) * (n+1)
            GUI.classAxes.plot([.7], [y], 'o', color=GUI.colors[n])

        #----------- Create buttons
        GUI.axbTrain = plt.axes([0.025, 0.45, 0.12, 0.075])
        GUI.axbRemove = plt.axes([0.025, 0.35, 0.12, 0.075])
        GUI.axbLrUp = plt.axes([0.025, 0.25, 0.12, 0.075])
        GUI.axbLrDn = plt.axes([0.025, 0.15, 0.12, 0.075])
        GUI.bTrain = Button(GUI.axbTrain, 'Train')
        GUI.bRemove = Button(GUI.axbRemove, 'Remove')
        GUI.bLrUp = Button(GUI.axbLrUp, 'Double LR')
        GUI.bLrDn = Button(GUI.axbLrDn, 'Halve LR')

        GUI.bTrain.on_clicked(GUI.funTrain)
        GUI.bRemove.on_clicked(GUI.funRemove)
        GUI.bLrUp.on_clicked(GUI.funLrUp)
        GUI.bLrDn.on_clicked(GUI.funLrDn)

    #============ define controllers, which connect between widgets and data
    def selectClass(className):
        GUI.currentLine = GUI.classNames.index(className)
        GUI.currentClass = className
        # cursor = Cursor(axes, useblit=True, color=colors[classNames.index(currentClass)], linewidth=0.5)
        # cursor.color = colors[classNames.index(currentClass)] # no work
        GUI.fig.canvas.draw()

    def predictRadioColor(mouseMoveEvent):
        if mouseMoveEvent.inaxes is not None and mouseMoveEvent.inaxes == GUI.classAxes:
            hovor =  GUI.find_radio_n(mouseMoveEvent.ydata)
            GUI.classRadios.activecolor = GUI.colors[hovor]
            # GUI.fig.suptitle("{:f}, {:f}".format(x, y))
            GUI.fig.canvas.draw()

    #------------ workarounds for stupid radio buttons, which yields no chances to control face color
    a = 0.9464; b = 0.0595
    def find_radio_n(radio_y):
        return len(GUI.colors)-int((radio_y-GUI.b)/(GUI.a-GUI.b)*len(GUI.colors))-1
    def find_radio_y(radio_n):
        return (-radio_n+1+len(GUI.colors)) * (GUI.a-GUI.b) / 7 + GUI.b

    def addDataPoint(buttonPressEvent):
        x, y = buttonPressEvent.xdata, buttonPressEvent.ydata
        if buttonPressEvent.inaxes is not None:
            addPoint = False
            for ax in GUI.axes:
                if buttonPressEvent.inaxes == ax:
                    addPoint = True
                    break
            if addPoint:
                GUI.x_list[GUI.currentLine].append(x)
                GUI.y_list[GUI.currentLine].append(y)
                for a in range(len(GUI.axes)):
                    GUI.lines2d[a][GUI.currentLine][0].set_data(GUI.x_list[GUI.currentLine], GUI.y_list[GUI.currentLine])
                if GUI.loss == "MCE":
                    GUI.showClassPredictions(np.array([[x,y]], dtype=float))
                GUI.showTitle(GUI.compileTitle(), "Point added.")
                GUI.fig.canvas.draw()

    def removePoints():
        GUI.x_list = [[] for c in GUI.colors]
        GUI.y_list = [[] for c in GUI.colors]
        GUI.refreshMainAxes(None)
        GUI.fig.canvas.draw()
        return

    #=========== External calls

    def refreshMainAxes(X, nContours=10, weights=None):
        # clear all axes.
        for ax in GUI.axes: ax.cla()
        # get model outputs, which are predictions.
        GUI.model.Feedforwards(GUI.XY).squeeze()
        outputs = GUI.model.GetOutputs()

        last_layer = len(outputs) - 1
        origin = 'lower'
        sLevels = (0, 1, 2, 3, 4, 5) if GUI.loss == 'MSE' else (0,)
        sColors = GUI.colors[: len(sLevels)]

        # Refresh prediction axes.
        for a in range(GUI.pred_dim):
            ax = GUI.axes[a]
            ax.set_xlim(-GUI.lim, GUI.lim)
            ax.set_ylim(-GUI.lim, GUI.lim)
            
            CS0 = ax.contourf(GUI.X, GUI.Y, outputs[last_layer][:,:,a], nContours, cmap=plt.cm.bone, origin=origin)
            CS = ax.contour(GUI.X, GUI.Y, outputs[last_layer][:,:,a], CS0.levels, colors=('k'), origin=origin, linewidths=.2)
            ax.contour(GUI.X, GUI.Y, outputs[last_layer][:,:,a], sLevels, colors=sColors, origin=origin, linewidths=.5)    
            plt.clabel(CS, fmt='%1.1f', colors='c', fontsize=8, inline=True)
            ax.plot([-GUI.lim, GUI.lim], [0, 0], '-', color='b', lw=0.2)
            ax.plot([0, 0], [-GUI.lim, GUI.lim], '-', color='b', lw=0.2)
            # if weights is not None:
            #     ax.plot([0, weights[0][last_layer][0] ], [0, weights[0][last_layer][1]], color='y')

        # Refresh partition ax if "MCE"
        if GUI.loss == "MCE":
            ax = GUI.axes[-1]
            ax.set_xlim(-GUI.lim, GUI.lim)
            ax.set_ylim(-GUI.lim, GUI.lim)
            # zeros = np.zeros_like(outputs[last_layer][:,:,a])
            for a in range(GUI.pred_dim):
                idx = [True] * GUI.pred_dim
                idx[a] = False
                surface = outputs[last_layer][:,:,a] - np.max(outputs[last_layer][:,:,idx], axis=-1)
                ax.contour(GUI.X, GUI.Y, surface, (0,), colors=GUI.colors[a], origin=origin, linewidths=.5)
                # surface = np.maximum(zeros, surface)
                # ax.contour(GUI.X, GUI.Y, surface, (0, 1), colors=(GUI.axcolor, GUI.colors[a]), origin=origin, linewidths=.5)

            
        # draw coordinate axes.
        for a in range(GUI.nAxes):
            GUI.lines2d[a] = [GUI.axes[a].plot(GUI.x_list[i], GUI.y_list[i], 'o', color=GUI.colors[i]) \
            for i in range(len(GUI.colors)) ]

        # put class prediction labels on data points
        if GUI.loss == "MCE" and X is not None:
            GUI.showClassPredictions(X)

        GUI.fig.canvas.draw()
        return

    def showClassPredictions(X):
        # put class prediction labels on data points
        pred = GUI.model.Feedforwards(X)
        prob = Layers.Softmax().Feedforwards(pred)
        cls = np.argmax(prob, axis=-1)
        prop = {'ha': 'center', 'va': 'center', 'color':'w', 'fontsize':8} #, 'bbox': {'fc': '0.8', 'pad': 0}}
        for a in range(len(GUI.axes)):
            for (x1, x2) in zip(X[cls==a, 0], X[cls==a, 1]):
                GUI.axes[a].text(x1, x2, str(a), prop, rotation=0)

    def activateDataCreator():
        GUI.classRadios.active = True
        for cursor in GUI.cursors:
            cursor.active = True
        GUI.classRadios.on_clicked(GUI.selectClass)
        GUI.mouseMoveBinding = GUI.fig.canvas.mpl_connect('motion_notify_event', GUI.predictRadioColor)
        GUI.mousePressBinding = GUI.fig.canvas.mpl_connect('button_press_event', GUI.addDataPoint)
        return

    def yieldControlToUI(extTrain, extRemove, extLrUp, extLrDn, compoleTitle):
        GUI.extTrain = extTrain
        GUI.extRemove = extRemove
        GUI.extLrUp = extLrUp
        GUI.extLrDn = extLrDn
        GUI.compileTitle = compoleTitle
        return

    def deactivateDataCreator():
        # classRadios.active = False
        for cursor in GUI.cursors:
            cursor.active = False
        GUI.fig.canvas.mpl_disconnect(GUI.mouseMoveBinding)
        GUI.fig.canvas.mpl_disconnect(GUI.mousePressBinding)
        return

    def showTitle(title, notification):
        (weightsShapes, samplesShape, activation, loss, learningRate) = title
        GUI.fig.suptitle("W.shape: " + str(weightsShapes) + ",   (X, Y).shape: " + str(samplesShape) \
            + "\nActivation: " + activation + ",   Loss: " + loss + ",   LR: " + str(learningRate) \
            + "\n" + notification, fontsize=10)
        plt.draw()

    def showPlot():
        plt.show()
        return

    def collectData():
        assert len(GUI.x_list) == len(GUI.y_list)
        x, y, classes = [], [], []
        for classId in range(len(GUI.x_list)):
            assert len(GUI.x_list[classId]) == len(GUI.y_list[classId])
            if len(GUI.x_list[classId]) > 0:
                x = x + GUI.x_list[classId]
                y = y + GUI.y_list[classId]
                classes = classes + [classId] * len(GUI.x_list[classId])
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        classes = np.array(classes, dtype=int)[..., np.newaxis]
        X = np.stack((x,y), axis=-1)
        Y = classes
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 2
        return X, Y

    def getRowsCols(n):
        rows = np.power(n, 0.5)
        cols = (n / rows)
        nRows, nCols = None, None
        if int(rows) < rows:
            if int(rows) * int(cols) >= rows * cols:
                nRows, nCols = int(rows), int(cols)
            else:
                if int(rows) * int(cols+1.) >= rows * cols:
                    nRows, nCols = int(rows), int(cols+1.)
                else:
                    nRows, nCols = int(rows+1.), int(cols+1.)
        else:
            nRows, nCols = int(rows), int(cols+.5)
            
        return nRows, nCols

    def flatten2DList(list2d):
        list2d = np.array(list2d)
        return list(list2d.flatten())

    def shareTicks(axesList, nRows, nCols):
        for a in range(len(axesList)):
            if a % nCols > 0:  # not a left ax
                axesList[a].sharey_foreign(axesList[0])
            if len(axesList) - a < nCols: # bottom ax
                axesList[a].sharex_foreign(axesList[max(0, a-nCols)])
    
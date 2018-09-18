from __future__ import division
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
from operator import itemgetter



#
# AnalyzeColon
#

class AnalyzeColon(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "AnalyzeColon" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# AnalyzeColonWidget
#

class AnalyzeColonWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input segmentation selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Segmentation: ", self.inputSelector)


    #
    # input cut point selector
    #
    self.cutPointSelector = slicer.qMRMLNodeComboBox()
    self.cutPointSelector.nodeTypes = ['vtkMRMLMarkupsFiducialNode']
    self.cutPointSelector.selectNodeUponCreation = True
    self.cutPointSelector.addEnabled = False
    self.cutPointSelector.removeEnabled = False
    self.cutPointSelector.noneEnabled = False
    self.cutPointSelector.showHidden = False
    self.cutPointSelector.showChildNodeTypes = False
    self.cutPointSelector.setMRMLScene(slicer.mrmlScene)
    self.cutPointSelector.setToolTip("Pick the cut points for the algorithm.")
    parametersFormLayout.addRow("Input Cut Points: ", self.cutPointSelector)


    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = AnalyzeColonLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.cutPointSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# AnalyzeColonLogic
#

class AnalyzeColonLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  # -------------- tool functions ---------------------------

  def findLocalMaximas(self, inList, minDist=0, threshold=0):
    '''A function to return a list of the local maximas of an input list. '''
    avgCurvature = np.mean(inList)
    localMaximas = []
    for x in range(1, len(inList) - 1):
      currentThreeList = inList[x - 1:x + 2]
      if currentThreeList[0] < currentThreeList[1] and currentThreeList[1] > currentThreeList[2] and currentThreeList[
        1] > avgCurvature * threshold:
        localMaximas.append((x + 1, currentThreeList[1]))
    # reprocess localMaximas:
    '''reprocessing will iterate through the list and find clusters of max points,
    where there is achain of points with less than minDist between them, and this procedss replaces this
    chain with a single point, in the middle of where the chain was. '''
    newLocalMaximas = []
    closePointsList = []
    addedOne = False
    for x, i in enumerate(localMaximas[:-1]):
      if not addedOne:
        if len(closePointsList) == 0:
          pass
        elif len(closePointsList) < 2:
          newLocalMaximas.append(closePointsList[0])
          closePointsList = []
        elif len(closePointsList) > 1:
          newLocalMaximas.append(closePointsList[len(closePointsList) // 2])
          closePointsList = []
      addedOne = False
      if closePointsList == []:
        closePointsList = [i]
      if localMaximas[x + 1][0] - i[0] < minDist:
        closePointsList.append(localMaximas[x + 1])
        addedOne = True
    newLocalMaximas.append(localMaximas[-1])
    return newLocalMaximas

  def findLocalMinimas(self, inList, minDist=0, threshold=0):
    '''A function to return a list of the local minimas of an input list. '''
    avgCurvature = np.mean(inList)
    localMinimas = []
    for x in range(1, len(inList) - 1):
      currentThreeList = inList[x - 1:x + 2]
      if currentThreeList[0] > currentThreeList[1] and currentThreeList[1] < currentThreeList[2] and currentThreeList[
        1] < avgCurvature / (threshold + 0.001):
        localMinimas.append((x + 1, currentThreeList[1]))
    # reprocess localMinimas:
    '''reprocessing will iterate through the list and find clusters of max points,
    where there is achain of points with less than minDist between them, and this procedss replaces this
    chain with a single point, in the middle of where the chain was. '''
    newLocalMinimas = []
    closePointsList = []
    addedOne = False
    for x, i in enumerate(localMinimas[:-1]):
      if not addedOne:
        if len(closePointsList) == 0:
          pass
        elif len(closePointsList) < 2:
          newLocalMinimas.append(closePointsList[0])
          closePointsList = []
        elif len(closePointsList) > 1:
          newLocalMinimas.append(closePointsList[len(closePointsList) // 2])
          closePointsList = []
      addedOne = False
      if closePointsList == []:
        closePointsList = [i]
      if localMinimas[x + 1][0] - i[0] < minDist:
        closePointsList.append(localMinimas[x + 1])
        addedOne = True
    newLocalMinimas.append(localMinimas[-1])
    return newLocalMinimas

  def unCluster(self, minList, maxList, curvatures):
    '''A function which takes a list of maximum points, a list of minimum points, and it looks for
    clusters of maximuns uninterupted with minimums, or vice versa. It replaces those clusters with a single
    point at the center of where the cluster was. Essentially, a better version of the reprocessing section of the
    in the find local maxima function. '''
    # print('In Whole Set: ' , curvatures)
    # print('In MIN: ' , minList)
    # print('In MAX: ' , maxList)
    newMinList = []
    newMaxList = []
    extremeList = []

    # make a sorted list of 3key tuples, where the last item identifies max/min
    for x in maxList:
      extremeList.append((x[0], x[1], 'MAX'))
    for x in minList:
      extremeList.append((x[0], x[1], 'MIN'))
    extremeList.sort(key=itemgetter(0))

    running = True
    count = 0
    while running:
      if count >= len(extremeList):
        break
      subList = [extremeList[count]]
      # look ahead as far as possible
      for x in range(1, len(curvatures)):
        # print(extremeList[count+x][2])
        # print(extremeList[count][2])
        # if the xth item after the first item is the same type (max/min)
        if count + x < len(extremeList) and extremeList[count + x][2] == extremeList[count][2]:
          subList.append(extremeList[count + x])
        else:  # if the next item is a different type (max/min)
          numberList = [int(z[0]) for z in subList]
          middleNumber = int(round(np.mean(numberList)))

          middlePoint = (middleNumber, curvatures[middleNumber - 1], extremeList[count][2])
          if middlePoint[2] == 'MAX':
            newMaxList.append(middlePoint)
          else:
            newMinList.append(middlePoint)
          count += len(subList) - 1
          # print(subList)
          break
      count += 1

    # remove the third item in the tuples
    newMinList = [(i[0], i[1]) for i in newMinList]
    newMaxList = [(i[0], i[1]) for i in newMaxList]
    # print('Out MIN: ', newMinList)
    # print('Out MAX: ', newMaxList)
    return newMinList, newMaxList

  def unitVector(self, vec):
    return vec / np.linalg.norm(vec)

  def angleBetween(self, v1, v2):
    v1U = self.unitVector(v1)
    v2U = self.unitVector(v2)
    return np.arccos(np.clip(np.dot(v1U, v2U), -1.0, 1.0))

  # ------------------- process functions -------------------------------------

  def convertSegmentationToBinaryLabelmap(self, segNode):
    segmentation = segNode.GetSegmentation()
    colSeg = None
    notColSeg = None

    for x in range(2):
      segment = segmentation.GetNthSegment(x)
      if len(segment.GetName()) < 7:
        colSeg = segment
      elif len(segment.GetName()) > 6:
        notColSeg = segment

    segmentation.RemoveSegment(notColSeg)

    colonBinLabelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
    slicer.mrmlScene.AddNode(colonBinLabelMapNode)
    slicer.vtkSlicerSegmentationsModuleLogic.ExportAllSegmentsToLabelmapNode(segNode, colonBinLabelMapNode)
    logging.info('Removed non colon, created binary labelmap.')
    return colonBinLabelMapNode


  def genCenterPoints(self, binLabelMapNode, tag='Sup'):
    pars = {}
    pars["InputImageFileName"] = binLabelMapNode.GetID()
    # create a markups fiducial node and name it, set it as the output
    fidsOut = slicer.vtkMRMLMarkupsFiducialNode()
    fidsOut.SetName('TEST0007_{}CenterPoints'.format(tag))
    slicer.mrmlScene.AddNode(fidsOut)
    pars["OutputFiducialsFileName"] = fidsOut.GetID()

    pars['NumberOfPoints'] = 600

    imgOut = slicer.vtkMRMLLabelMapVolumeNode()
    imgOut.SetName('TEST0007_OutputImg')
    slicer.mrmlScene.AddNode(imgOut)
    pars['OutputImageFileName'] = imgOut.GetID()
    logging.info('Created pars, running extract skeleton')

    # run the module with parameters
    extractor = slicer.modules.extractskeleton
    slicer.cli.runSync(extractor, None, pars)
    logging.info('Extracted Skeleton.')
    return fidsOut

  def fitCurve(self, fidsNode):
    '''A function to return a curve model from a fiducial list node
    with specific parameters for this project. '''
    logging.info('Fitting curve to center points...')
    markupsToModelNode = slicer.vtkMRMLMarkupsToModelNode()
    markupsToModelNode.SetName('MyMarkupsToModelNode')
    slicer.mrmlScene.AddNode(markupsToModelNode)

    markupsToModelNode.SetAndObserveInputNodeID(fidsNode.GetID())

    outputCurveNode = slicer.vtkMRMLModelNode()
    slicer.mrmlScene.AddNode(outputCurveNode)
    outputCurveNode.SetName(fidsNode.GetName()[:-17] + 'Curve')

    markupsToModelNode.SetAndObserveModelNodeID(outputCurveNode.GetID())
    markupsToModelNode.SetModelType(1)
    markupsToModelNode.SetModelType(1)
    markupsToModelNode.SetCurveType(3)
    markupsToModelNode.SetPolynomialFitType(1)
    markupsToModelNode.SetPolynomialOrder(2)
    markupsToModelNode.SetPolynomialSampleWidth(0.05)
    markupsToModelNode.SetTubeRadius(0)
    logging.info('Fitted curve to center points. ')

    return outputCurveNode


  def makeCurvaturesFile(self, curveNode):
    logging.info("Computing and writing curvatures...")
    polyData = curveNode.GetPolyData()
    import CurveMaker
    CurveMaker.CurveMakerLogic()
    curvatureArray = vtk.vtkDoubleArray()

    avgCurve, minCurve, maxCurve = CurveMaker.CurveMakerLogic().computeCurvatures(polyData, curvatureArray)

    polyData.GetPointData().AddArray(curvatureArray)
    curveDisplayNode = curveNode.GetDisplayNode()

    curveDisplayNode.SetActiveScalarName('Curvature')
    curveDisplayNode.SetScalarVisibility(1)

    curvatureList = [curvatureArray.GetTuple1(x) for x in range(curvatureArray.GetNumberOfTuples())]
    stringCurvatureList = [str(x) for x in curvatureList]

    points = polyData.GetPoints()
    pointList = [points.GetPoint(x) for x in range(points.GetNumberOfPoints())]

    newLines = [stringCurvatureList[x] + ', ' + str(pointList[x][0]) + ', ' + str(pointList[x][1]) + ', ' + str(
      pointList[x][2]) + '\n' for x in range(len(pointList))]
    newLines.append("\n")

    outPath = r"C:\Users\jaker\Documents\Curvatures.txt"
    print(outPath)
    outFile = open(outPath, 'w')
    outFile.writelines(newLines)
    outFile.close()
    logging.info("Saved curvatures.")

  def makeCutPointsFile(self, fidsNode):
    outPath = r"C:\Users\jaker\Documents\CutPoints.txt"
    fOut = open(outPath, 'w')
    for x in range(2):
      pos = np.zeros(3)
      fidsNode.GetNthFiducialPosition(x, pos)
      fOut.write('{},{},{}\n'.format(pos[0], pos[1], pos[2]))
    fOut.close()

  def addDetails(self, inPath, outPath): # TODO test that this works
    '''A function that takes the path of a text file and creates a new text file with the point number,
        and the percentage of how far the point is along the list. Easy to import to Excel'''
    inFile = open(inPath, 'r')
    lines = inFile.readlines()
    inFile.close()
    lines = [x.strip().split(', ') for x in lines]
    outFile = open(outPath, 'w')
    outFile.write(inPath[-26:] + "\n")
    for count, item in enumerate(lines):
        if item != '' and item != "\n" and item != ['']:
            outFile.write(
                '{}, {}, {}, {}, {}, {}'.format(count,  100 * count / (len(lines) - 2), item[1], item[2], item[3], # TODO the % of length stat is not working, is zero every time it was giving jut  integers, not float.
                                                item[0]) + '\n') # it was count / (len(lines) - 2) * 100
    outFile.close()

  def getSumCurvatures(self, curvaturesList, width): # TODO test that this works
    '''A function which takes a list, and it returns a new list of equal length, where each value corresponds
    to the same indexed value in the first list, plus the all the items 'width' positions up and down the list.'''
    sumList = []
    for x in range(len(curvaturesList)):
      subList = (curvaturesList[max(x - width, 0): min(x + width + 1, len(curvaturesList))])
      subList = [float(y) for y in subList]
      sumList.append(sum(subList))
    return sumList

  def addSumCurvaturesToDataFile(self, inPath, width=10): # TODO test that this works
      '''A fucntion to modify a detailed data file with curvatures, by adding a column
      that contains the sum of curvatures in a given interval for every point '''
      fIn = open(inPath, 'r')
      lines = fIn.readlines()
      fIn.close()
      title = lines[0].strip()

      curvatureValues = [x.strip().split(', ')[5] for x in lines[1:]]
      sumCurvatureValues = self.getSumCurvatures(curvatureValues, width)
      newLines = [title] + [lines[x].strip() + ', ' + str(sumCurvatureValues[x - 1]) for x in
                            range(1, len(sumCurvatureValues) + 1)]
      fOut = open(inPath, 'w')
      for line in newLines:
          fOut.write(line + '\n')
      fOut.close()

  def addSumCurvatureMaxMinsToDataFile(self, inPath, minPointDist=0, threshold=1, minThresholdBoost=1.5):
    '''A function to add a column to the data file whihc indicates if the point is at a max or a min. '''
    fIn = open(inPath, 'r')
    lines = fIn.readlines()
    fIn.close()
    title = lines[0].strip()
    sumCurvatureValues = [x.strip().split(', ')[6] for x in lines[1:]]
    sumCurvatureValues = [float(y) for y in sumCurvatureValues]
    locMaximas = self.findLocalMaximas(sumCurvatureValues, minPointDist, threshold)
    locMinimas = self.findLocalMinimas(sumCurvatureValues, minPointDist,
                                  threshold * minThresholdBoost)  # The minimas are currently being held at a higher threshold so the maximas are unClustered more.

    # print('Calling unCluster!')
    locMinimas, locMaximas = self.unCluster(locMinimas, locMaximas, sumCurvatureValues)

    # xVals = [x.strip().split(', ')[0] for x in lines[1:]]
    # xVals = [int(x) for x in xVals]
    locExtremesColumn = []
    for x in range(1, len(lines)):
      t = (x, sumCurvatureValues[x - 1])
      if t in locMaximas:
        locExtremesColumn.append('MAX')
      elif t in locMinimas:
        locExtremesColumn.append('MIN')
      else:
        locExtremesColumn.append('0')
    newLines = [title] + [lines[x].strip() + ', ' + str(locExtremesColumn[x - 1]) for x in
                          range(1, len(locExtremesColumn) + 1)]
    fOut = open(inPath, 'w')
    for line in newLines:
      fOut.write(line + '\n')
    fOut.close()


  def addDegreeChangesToFile(self, inPath):
    '''This function look at every max, and makes a sublist of itself, and the minimums on either side.
    it then akes a tangent at each minimum, and compares the change in angle from one to the other,
    saying that the line curves that x degrees over the straight line distance from Min to Min. '''
    fIn = open(inPath, 'r')
    lines = fIn.readlines()
    fIn.close()
    title = lines[0].strip()
    curvatureValues = [x.strip().split(', ')[5] for x in lines[1:]]
    numVals = [x.strip().split(', ')[0] for x in lines[1:]]
    maxMinTypes = [x.strip().split(', ')[7] for x in lines[1:]]
    coords = [(x.strip().split(', ')[2], x.strip().split(', ')[3], x.strip().split(', ')[4]) for x in lines[1:]]

    extremePoints = []
    maxPlaces = []
    for y in range(len(lines) - 1):
      if maxMinTypes[y] == 'MAX' or maxMinTypes[y] == 'MIN':
        extremePoints.append((coords[y][0], coords[y][1], coords[y][2], maxMinTypes[y], numVals[y]))

    angleChangeList = []
    for x in range(1, len(extremePoints) - 1):
      subList = [extremePoints[x - 1], extremePoints[x], extremePoints[x + 1]]

      if subList[1][3] == 'MAX':
        leftMinForwardPointNum = int(subList[0][4]) + 2
        rightMinBackwardPointNum = int(subList[2][4]) - 2
        leftMinForwardPointCoords = np.array(
          [float(coords[leftMinForwardPointNum][0]), float(coords[leftMinForwardPointNum][1]),
          float(coords[leftMinForwardPointNum][2])])
        rightMinBackwardPointCoords = np.array(
          [float(coords[rightMinBackwardPointNum][0]), float(coords[rightMinBackwardPointNum][1]),
          float(coords[rightMinBackwardPointNum][2])])

        vecPosList = [np.array([float(z[0]), float(z[1]), float(z[2])]) for z in subList]

        vecOne = leftMinForwardPointCoords - vecPosList[0]
        vecTwo = vecPosList[2] - rightMinBackwardPointCoords

        vecThree = vecPosList[2] - vecPosList[0]
        angleChange = self.angleBetween(vecOne, vecTwo) * 180 / np.pi
        straightDist = np.linalg.norm(vecThree)

        angleChangeList.append((subList[1][4], angleChange, straightDist))

    # print(angleChangeList)
    angleChangeValues = []
    straightDistValues = []
    numList = [int(x[0]) for x in angleChangeList]

    for i in range(len(lines) - 1):
      if i in numList:
        for x in angleChangeList:
          if int(x[0]) == i:
            angleChangeValues.append(x[1])
            straightDistValues.append(x[2])
      else:
        angleChangeValues.append('0')
        straightDistValues.append('0')

    newLines = [title] + [
      lines[x].strip() + ', ' + str(angleChangeValues[x - 1]) + ', ' + str(straightDistValues[x - 1]) for x in
      range(1, len(angleChangeValues) + 1)]
    fOut = open(inPath, 'w')
    for line in newLines:
      fOut.write(line + '\n')
    fOut.close()



  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def run(self, inputSegmentation, inputCutPoints, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputSegmentation, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    self.binLabelMapNode = self.convertSegmentationToBinaryLabelmap(inputSegmentation)
    self.centerPointsNode = self.genCenterPoints(self.binLabelMapNode, 'Sup')
    self.curveNode = self.fitCurve(self.centerPointsNode)
    self.makeCurvaturesFile(self.curveNode)
    self.makeCutPointsFile(inputCutPoints) #working to here
    self.addDetails(r"C:\Users\jaker\Documents\Curvatures.txt", r"C:\Users\jaker\Documents\CurvaturesData.txt")
    self.addSumCurvaturesToDataFile(r"C:\Users\jaker\Documents\CurvaturesData.txt", 0) # TODO fix hardcode
    self.addSumCurvatureMaxMinsToDataFile(r"C:\Users\jaker\Documents\CurvaturesData.txt", 0, 1, 1.5)
    self.addDegreeChangesToFile(r"C:\Users\jaker\Documents\CurvaturesData.txt")

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('AnalyzeColonTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class AnalyzeColonTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_AnalyzeColon1()

  def test_AnalyzeColon1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = AnalyzeColonLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')

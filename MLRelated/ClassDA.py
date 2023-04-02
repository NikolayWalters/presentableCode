from astropy.io import fits
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
import pickle
from argparse import ArgumentParser
import time
from tqdm import tqdm

def fni(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def list_fits(directory):
    fileList = []
    if directory.endswith('/') == False:
        directory = directory + '/'
    for file in os.listdir(directory):
        if file.endswith(".fits"):
            f = os.path.join(directory, file)
            fileList.append(f)
    if not fileList:
        raise Exception('No .fits found in the directory')
    listLength = len(fileList)
    print('Found '+str(listLength)+' .fits')
    return fileList, listLength

def splitter(fileList, listLength):
    partitioned = False
    if listLength > 1000:
        partitions = [fileList[x:x+1000] for x in range(0, listLength, 1000)]
        partitioned = True
        print('Partitioned into '+str(len(partitions))+' batches')
    else:
        partitions = fileList
        print('No partition needed')     
    return partitioned, partitions

def preProcess(element):
    tempData = []
    hdul = fits.open(element)
    typeEx = hdul[0].data
    dataF = typeEx[0,:]
    dataWave = typeEx[2,:]
    hdul.close()
    start = fni(dataWave, 3838.839)
    finish = fni(dataWave, 8914.597)
    tempData = dataF[start:finish]
    GaussData = gaussian_filter(tempData, sigma=1)
    GaussMax = np.max(GaussData)
    GaussMin = np.min(GaussData)
    for n, i in enumerate(tempData):
        if i > GaussMax:
            tempData[n] = GaussMax
        if i < GaussMin:
            tempData[n] = GaussMin
    tempData = (np.array(tempData) - np.mean(tempData))/np.std(tempData)
    return tempData, dataWave

def plots(name, points):
    pXMesh = np.linspace(3837.956, 8914.597, 3659)
    plt.title(name)
    plt.plot(pXMesh, points, 'k-', lw=0.2)
    plt.xlabel('Wavelength, A')
    plt.ylabel('Relative Flux')
    plt.show()

def sPlots(name, points, direct):
    pXMesh = np.linspace(3837.956, 8914.597, 3659)
    plt.title(name)
    plt.plot(pXMesh, points, 'k-', lw=0.2)
    plt.xlabel('Wavelength, A')
    plt.ylabel('Relative Flux')
    misName = name.replace('.fits', '')
    savePath = direct + 'plots/' + misName + '.png'
    try:
        plt.savefig(savePath, dpi=300)
    except FileNotFoundError:
        toMake = directory + 'plots/'
        os.makedirs(toMake)
        plt.savefig(savePath, dpi=300)
    plt.close()

def qso(element):
    hdul = fits.open(element)
    typeEx = hdul[0].header['CLASS']
    isQSO = False
    if typeEx == 'QSO':
        isQSO = True
    hdul.close()
    return isQSO

def classify(partitioned, partitions, model, directory, display=False, savePlot=False, saveOut=False):
    pXMesh = np.linspace(3837.956, 8914.597, 3659)
    globalMisclass = []
    if partitioned: 
        for k in tqdm(range(len(partitions))):
        #for k in tqdm(range(4)): used for testing
            count = 0
            names = []
            for el in partitions[k]:
                name = el.replace(directory, '')
                names.append(name)
                tempData, dataWave = preProcess(el)
                if count == 0:
                    dataStackN = tempData
                    minWave = np.min(dataWave)
                    maxWave = np.max(dataWave)
                else:
                    dataStackN = np.vstack((dataStackN, tempData))
                    if np.min(dataWave) > minWave:
                        minWave = np.min(dataWave)
                    if np.max(dataWave) < maxWave:
                        maxWave = np.max(dataWave)
                count = 1
            NoTars = np.zeros(np.shape(dataStackN)[0])
            y_pred = model.predict(dataStackN)
            misclas = []
            label = []
            j = 0
            for j in range(len(NoTars)):
                if NoTars[j] != y_pred[j]:
                    misName = names[j].replace('/', '')
                    toQSO = directory + misName
                    isQSO = qso(toQSO)
                    if isQSO:
                        print('QSO skipped: ', misName)
                        continue
                    misclas.append(misName)
                    label.append(NoTars[j])
                    if display:
                        plots(names[j], dataStackN[j][:])
                    if savePlot:
                        sPlots(names[j], dataStackN[j][:], directory)
                    if misclas and len(globalMisclass) == 0:
                        globalMisclass = misclas
                    else:
                        globalMisclass = np.concatenate((globalMisclass, misclas))
    else:
        names = []
        count = 0
        for el in partitions:
            name = el.replace(directory, '')
            names.append(name)
            tempData, dataWave = preProcess(el)
            if count == 0:
                dataStackN = tempData
                minWave = np.min(dataWave)
                maxWave = np.max(dataWave)
            else:
                dataStackN = np.vstack((dataStackN, tempData))
                if np.min(dataWave) > minWave:
                    minWave = np.min(dataWave)
                if np.max(dataWave) < maxWave:
                    maxWave = np.max(dataWave)
            count = 1
        NoTars = np.zeros(np.shape(dataStackN)[0])
        y_pred = model.predict(dataStackN)
        misclas = []
        label = []
        j = 0
        for j in range(len(NoTars)):
            if NoTars[j] != y_pred[j]:
                misName = names[j].replace('/', '')
                toQSO = directory + misName
                isQSO = qso(toQSO)
                if isQSO:
                    print('QSO skipped: ', misName)
                    continue
                misclas.append(misName)
                label.append(NoTars[j])
                if display:
                    plots(names[j], dataStackN[j][:])
                if savePlot:
                    sPlots(names[j], dataStackN[j][:], directory)
                if misclas and len(globalMisclass) == 0:
                    globalMisclass = misclas
                else:
                    globalMisclass = np.concatenate((globalMisclass, misclas))
    if saveOut:
        pathOutput = directory + 'IdentifiedDAs'
        with open(pathOutput, 'w') as f:
            for item in globalMisclass:
                f.write("%s\n" % item)
    print('Identified potential DAs:')
    if len(globalMisclass) == 0:
        print('None')
    else:
        print(globalMisclass)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Specify appropriate directory for identification')
    parser.add_argument('data_directory',
                        help='Directory path that contains .fits to be identified')
    parser.add_argument('-sp', '--saveplots', action='store_true',
                        help='when enabled via "--saveplots" or "-sp" will save diagrams')
    parser.add_argument('-sl', '--savelist', action='store_true',
                        help='when enabled via "--savelist" or "-sl" will save identified DAs as a list')
    parser.add_argument('-dp', '--displayplots', action='store_true',
                        help='when enabled via "--displayplots" or "-dp" will display diagrams')
    arguments = parser.parse_args()
    saveplots = False
    savelist = False
    displayplots = False
    target_directory = arguments.data_directory
    if target_directory[-1] != '/':
        target_directory = target_directory + '/'
    if arguments.saveplots:
        saveplots = True
        print('Plots will be saved in ' + target_directory + 'plots/\n')
    if arguments.savelist:
        savelist = True
        print('List of potential DAs will be saved as ' + target_directory + 'IdentifiedDAs\n')
    if arguments.displayplots:
        displayplots = True
        print('Individual plots will be displayed sequentialy\n')
    fileList, listLength = list_fits(target_directory)
    partitioned, partitions = splitter(fileList, listLength)

    home = os.path.dirname(os.path.realpath(__file__))
    home = home + '/clf_comb.p'
    try:
        model = pickle.load(open(home, "rb"))
    except FileNotFoundError:
        print('Model not found. Model must be in the same directory as the script and named "clf_comb.p".')
        sys.exit()
    classify(partitioned, partitions, model, target_directory, displayplots, saveplots, savelist)
    print('Done')
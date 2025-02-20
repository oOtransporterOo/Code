# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:30:37 2025

@author: Silas Hauser
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import inspect
import pandas as pd
degreeOfRegression = 3
trainedCoef= []

subsetOfVariables = ["LotArea","YearBuilt","YearRemodAdd","1stFlrSF","2ndFlrSF","GrLivArea","BedroomAbvGr","Fireplaces",
                     "PoolArea","GarageArea", "GarageFinish","MSZoning","Street","Alley","LandContour","LotShape", "Utilities",
                     "LandSlope","Neighborhood","Condition1", "Condition2","BldgType", "HouseStyle" ,"OverallQual","OverallCond",
                     "RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation",
                     "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinSF2","TotalBsmtSF","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical",
                     "LotFrontage","LotConfig","LowQualFinSF","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","KitchenAbvGr","TotRmsAbvGrd","GarageYrBlt","GarageCars",
                     "WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","MiscVal","MoSold","BsmtUnfSF","ExterQual", "ExterCond", 
                     "Foundation", "BsmtQual", "BsmtCond","BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 
                     "CentralAir", "Electrical", "LotConfig", "KitchenQual", "Functional","FireplaceQu", "GarageType", "GarageQual", "GarageCond", 
                     "PavedDrive","PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
setOfSpecialVariables = ["GarageFinish","MSZoning","Street","Alley","LandContour","LotShape","Utilities","LandSlope","Neighborhood", 
                         "Condition1", "Condition2","BldgType", "HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType",
                         "ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical",
                         "LotConfig","ExterQual", "ExterCond", 
                         "Foundation", "BsmtQual", "BsmtCond","BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", 
                         "CentralAir", "Electrical", "LotConfig", "KitchenQual", "Functional","FireplaceQu", "GarageType", "GarageQual", "GarageCond", 
                         "PavedDrive","PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]


def convertCategorical(data, categories, feature_name):
    returnData = []
    irregularCounter = 0
    mapping = {category: idx for idx, category in enumerate(categories)}
    
    for x in data:
        encoding = [0] * len(categories)
        if x in mapping:
            encoding[mapping[x]] = 1
        else:
            irregularCounter += 1
        returnData.append(encoding)
    if irregularCounter > len(data) * 0.05:
        print("Irregular Trainingset:", feature_name,"percentage:", irregularCounter/len(data))
    
    return np.array(returnData).T

def convertCondition1(data):
    categories = ["Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertCondition2(data):
    categories = ["Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertBldgType(data):
    categories = ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertHouseStyle(data):
    categories = ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertRoofStyle(data):
    categories = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertRoofMatl(data):
    categories = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertExterior1st(data):
    categories = ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertExterior2nd(data):
    categories = ["AsbShng", "AsphShn", "BrkComn", "Brk Cmn", "BrkFace", "CBlock", "CmentBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng","Wd Shng", "WdShing"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertMasVnrType(data):
    categories = ["BrkCmn", "BrkFace", "CBlock", "None", "Stone"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertExterQual(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertExterCond(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertFoundation(data):
    categories = ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertBsmtQual(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertBsmtCond(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertBsmtExposure(data):
    categories = ["Gd", "Av", "Mn", "No", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertBsmtFinType1(data):
    categories = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertBsmtFinType2(data):
    categories = ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertHeating(data):
    categories = ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertHeatingQC(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertCentralAir(data):
    categories = ["N", "Y"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertElectrical(data):
    categories = ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertLotConfig(data):
    categories = ["Inside", "Corner", "CulDSac", "FR2", "FR3"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertKitchenQual(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertFunctional(data):
    categories = ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertFireplaceQu(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertGarageType(data):
    categories = ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertGarageQual(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertGarageCond(data):
    categories = ["Ex", "Gd", "TA", "Fa", "Po", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertPavedDrive(data):
    categories = ["Y", "P", "N"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertPoolQC(data):
    categories = ["Ex", "Gd", "TA", "Fa", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertFence(data):
    categories = ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertMiscFeature(data):
    categories = ["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertSaleType(data):
    categories = ["WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertSaleCondition(data):
    categories = ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"]
    return convertCategorical(data, categories, inspect.currentframe().f_code.co_name)

def convertNeighborhood(data):
    returnData = []
    irregularCounter = 0    
    # Mapping Neighborhood names to indices in the one-hot encoding list
    neighborhood_mapping = {
        "Blmngtn": 0,
        "Blueste": 1,
        "BrDale": 2,
        "BrkSide": 3,
        "ClearCr": 4,
        "CollgCr": 5,
        "Crawfor": 6,
        "Edwards": 7,
        "Gilbert": 8,
        "IDOTRR": 9,
        "MeadowV": 10,
        "Mitchel": 11,
        "NAmes": 12,
        "NoRidge": 13,
        "NPkVill": 14,
        "NridgHt": 15,
        "NWAmes": 16,
        "OldTown": 17,
        "SWISU": 18,
        "Sawyer": 19,
        "SawyerW": 20,
        "Somerst": 21,
        "StoneBr": 22,
        "Timber": 23,
        "Veenker": 24
    }
    
    # Iterate through the data and apply one-hot encoding
    for x in data:
        # Initialize a list with zeros for each neighborhood
        encoding = [0] * len(neighborhood_mapping)
        
        if x in neighborhood_mapping:
            encoding[neighborhood_mapping[x]] = 1
            returnData.append(encoding)
        else:
            irregularCounter += 1
            returnData.append([0] * len(neighborhood_mapping))
    
    # Print warning if more than 5% of the entries are irregular
    if irregularCounter > len(data) * 0.05:
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))  
    
    # Return the encoded data transposed (for consistency)
    return np.array(returnData).T


def convertLandSlope(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "Gtl":
            returnData.append([1,0,0])
        elif x == "Mod":
            returnData.append([0,1,0])
        elif x == "Sev":
            returnData.append([0,0,1])
        else:
            irregularCounter+=1
            returnData.append([0,0,0])
    
    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))  
    return np.array(returnData).T
    

def convertUtilities(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "AllPub":
            returnData.append([1,0,0,0])
        elif x == "NoSewr":
            returnData.append([0,1,0,0])
        elif x == "NoSeWa":
            returnData.append([0,0,1,0])
        elif x == "ELO":
            returnData.append([0,0,0,1])
        else:
            irregularCounter+=1
            returnData.append([0,0,0,0])
    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))        
    return np.array(returnData).T


def convertLotShape(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "Reg":
            returnData.append([1,0,0,0])
        elif x == "IR1":
            returnData.append([0,1,0,0])
        elif x == "IR2":
            returnData.append([0,0,1,0])
        elif x == "IR3":
            returnData.append([0,0,0,1])
        else:
            irregularCounter+=1
            returnData.append([0,0,0,0])

    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))          
    return np.array(returnData).T

def convertLandContour(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "Lvl":
            returnData.append([1,0,0,0])
        elif x == "Bnk":
            returnData.append([0,1,0,0])
        elif x == "HLS":
            returnData.append([0,0,1,0])
        elif x == "Low":
            returnData.append([0,0,0,1])
        else:
            irregularCounter+=1
            returnData.append([0,0,0,0])

    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))          
    return np.array(returnData).T

def convertGarageFinish(data):
    returnData = []
    irregularCounter = 0
    for x in data:
        if x == "Fin":
            returnData.append(2)
        elif x == "RFn":
            returnData.append(1)
        elif x == "Unf":
            returnData.append(-2)
        elif x == "NA":
            returnData.append(0)
        else:
            returnData.append(0) 
            irregularCounter+=1
    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))  
    return returnData
                
def convertMSZoning(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "A":
            returnData.append([1,0,0,0,0,0,0,0])
        elif x == "C":
            returnData.append([0,1,0,0,0,0,0,0])
        elif x == "FV":
            returnData.append([0,0,1,0,0,0,0,0])
        elif x == "I":
            returnData.append([0,0,0,1,0,0,0,0])
        elif x == "RH":
            returnData.append([0,0,0,0,1,0,0,0])
        elif x == "RL":
            returnData.append([0,0,0,0,0,1,0,0])
        elif x == "RP":
            returnData.append([0,0,0,0,0,0,1,0])
        elif x == "RM":
            returnData.append([0,0,0,0,0,0,0,1])
        else:
            returnData.append([0,0,0,0,0,0,0,0])
            irregularCounter+=1

    
    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))  
    return np.array(returnData).T

def convertStreet(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "Grvl":
            returnData.append([1,0]) 
        elif x == "Pave":
            returnData.append([0,1])
        else:
            returnData.append([0,0])
    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))  
    return np.array(returnData).T

def convertAlley(data):
    returnData = []
    irregularCounter = 0;
    for x in data:
        if x == "Grvl":
            returnData.append([1,0]) 
        elif x == "Pave":
            returnData.append([0,1])
        else:
            returnData.append([0,0])
    if(irregularCounter>len(data)*0.05):
        print("Irregular Trainingset: ", inspect.currentframe().f_code.co_name, "percentage:", irregularCounter/len(data))  
    return np.array(returnData).T

def getSubsetOfData(headerNames=subsetOfVariables):
    data = readCSV()
    subsetData = []
    for x in headerNames:
        if(x in setOfSpecialVariables):
            if x == "GarageFinsih":        
                subsetData.append(convertGarageFinish(data[x]))    
            elif x == "MSZoning":
                for x in convertMSZoning(data[x]):
                    subsetData.append(x)
            elif x == "Street":
                for x in convertStreet(data[x]):
                    subsetData.append(x)
            elif x == "Alley":
                for x in convertAlley(data[x]):
                    subsetData.append(x)
            elif x == "LotShape":
                for x in convertLotShape(data[x]):
                    subsetData.append(x)
            elif x == "LandContour":
                for x in convertLandContour(data[x]):
                    subsetData.append(x)
            elif x == "Utilities":
                for x in convertUtilities(data[x]):
                    subsetData.append(x)
            elif x == "LandSlope":
                for x in convertLandSlope(data[x]):
                    subsetData.append(x)
            elif x == "Neighborhood":
                for x in convertNeighborhood(data[x]):
                    subsetData.append(x)
            elif x == "Condition1":
                for x in convertCondition1(data[x]):
                    subsetData.append(x)
            elif x == "Condition2":
                for x in convertCondition2(data[x]):
                    subsetData.append(x)
            elif x == "BldgType":
                for x in convertBldgType(data[x]):
                    subsetData.append(x)
            elif x == "HouseStyle":
                for x in convertHouseStyle(data[x]):
                    subsetData.append(x)
            elif x == "RoofStyle":
                for x in convertRoofStyle(data[x]):
                    subsetData.append(x)
            elif x == "RoofMatl":
                for x in convertRoofMatl(data[x]):
                    subsetData.append(x)
            elif x == "Exterior1st":
                for x in convertExterior1st(data[x]):
                    subsetData.append(x)
            elif x == "Exterior2nd":
                for x in convertExterior2nd(data[x]):
                    subsetData.append(x)
            elif x == "MasVnrType":
                for x in convertMasVnrType(data[x]):
                    subsetData.append(x)
                            
                    
        else:
            correctedData = verifyValues(data[x])
            subsetData.append(pd.Series(correctedData))
    return np.array(subsetData).T

def verifyValues(data):
    int_values = [x for x in data if isinstance(x, int)]

    if not int_values:  # Avoid division by zero if no integers exist
        mean_value = 0
    else:
        mean_value = int(round(np.mean(int_values)))  # Compute rounded mean

    return [x if isinstance(x, int) else mean_value for x in data]

def getCSVHeaders():
    data =readCSV()
    return data.columns

def readCSV(filePath="train.csv"):
    data = pd.read_csv(filePath)
    return data

def plotPredictions(predictions,trueValues, x):

    plt.scatter(x, trueValues, s=50,c="blue",marker='H',label="True Values")
    plt.scatter(x, predictions, s=20,c="red",alpha=0.75, label = "Prediction")

 
    # plt.axvline(x=10,color="black",linestyle="--")
    titleValues = " \n ".join([" ; ".join(subsetOfVariables[i:i+6]) for i in range(0,len(subsetOfVariables),6)])
    plt.title( titleValues, fontsize=10)
    plt.legend()
    
    plt.show()
    
def generateRandomSet(setSize=10,lowerRange=0,upperRange=40):
    random_list = [[random.randint(lowerRange, upperRange)for _ in range(4)] for _ in range(setSize)]
    return random_list

# def testCoeff(coef,testSet):
#     prediction = predict_polynomial(coef, np.array(testSet))
#     trueValue= myTestFunction(testSet)
#     print(prediction,trueValue)
#     plotPredictions(prediction, trueValue)

def myTestFunction(myList):
    returnList=[]
    for x in myList:
        returnList.append( 4*x[0]-2.5*x[1]+2*x[2]**2+0.01*x[3]+470)
    return returnList    
    

# Function to create polynomial features for multiple input variables
def create_polynomial_features(X, degree):

    n_samples, n_features = X.shape
    features = []
    
    # Generate polynomial terms for each feature and degree
    for d in range(degree + 1):  # From x^0 to x^degree
        for feature_index in range(n_features):
            features.append(X[:, feature_index] ** d)  # Add polynomial terms for each feature
    
    return np.column_stack(features)

# Function to fit polynomial regression model with Ridge Regularization (L2 Regularization)
def fit_polynomial_ridge(X, y, degree=degreeOfRegression, lambda_reg=0.1):
   

    # Create polynomial features for the input data
    X_poly = create_polynomial_features(X, degree)
    
    # Add regularization term (Ridge Regression)
    X_transpose = X_poly.T
    identity_matrix = np.eye(X_poly.shape[1])  # Identity matrix for regularization
    identity_matrix[0, 0] = 0  # Don't regularize the intercept term
    
    # Ridge Regression solution: (X^T * X + lambda * I)^(-1) * X^T * y
    coeffs = np.linalg.inv(X_transpose @ X_poly + lambda_reg * identity_matrix) @ (X_transpose @ y)
    
    return coeffs

# Function to make predictions using the trained coefficients
def predict_polynomial(coeffs, X, degree=degreeOfRegression):
  
    # Create polynomial features for the input data
    X_poly = create_polynomial_features(X, degree)
    
    # Calculate predictions: dot product of polynomial features and coefficients
    predictions = np.dot(X_poly, coeffs)
    
    return predictions

def buildModel(start=0,end=-1):
    trainedCoef.append(fit_polynomial_ridge(getSubsetOfData()[start:end:],readCSV()["SalePrice"][start:end:]))
    
def createPlot(start = 0, stepsize = 1,inputData = getSubsetOfData()):
    
    plotPredictions(predict_polynomial(trainedCoef[0], inputData)[start::stepsize], readCSV()["SalePrice"][start::stepsize], readCSV()["Id"][start::stepsize])
    
def run(startPlot = 0,stepsizePlot = 1, startTrainData=0, endTrainData=-1):
    trainedCoef.clear()
    buildModel(start=startTrainData,end=endTrainData)
    createPlot(startPlot,stepsizePlot)

# Example usage
if __name__ == "__main__":
    print("---")
    # X_train = generateRandomSet(100)
    # y_train = myTestFunction(X_train)
    
    # trainedCoef = fit_polynomial_ridge(np.array(X_train), np.array(y_train))


    # X_test = generateRandomSet(25,0,10)
    
    

    



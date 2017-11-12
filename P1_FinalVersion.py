import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import time
import pandas as pd
import progressbar
from sympy import prevprime


def convert_matrix(my_data):
    nmovies = np.amax(my_data, axis=0)[1]
    nusers = np.amax(my_data, axis=0)[0]
    print "Users", nusers, "Movies", nmovies
    df = np.zeros((nmovies, nusers),dtype=np.bool)
    for i in range(int(np.shape(my_data)[0])):
        if my_data[i, 2] > 2.5:
            df[my_data[i, 1] - 1, my_data[i, 0] - 1] = 1
    return df



def createSigMat(mat, nusers, nmovies,nhash):
    largestPrime = 29009
    a = np.random.randint(largestPrime, size=(nhash,1))
    b = np.random.randint(largestPrime, size=(nhash, 1))
    sigMat = np.zeros((nhash, nusers),dtype=np.int32) # number of hash functions
    sigMat.fill(999999) # fill with high number
    bar = progressbar.ProgressBar()
    loop_time = time.time()

    for user in bar(range(nusers)):
        movies = mat[user][0]
        movies = np.array([movies])
        if movies.shape[1] > 0:
            newMat = np.dot(a,movies)
            resArray = (newMat + b) % largestPrime
            hashVal = np.amin(resArray,axis=1)
            sigMat[:, user] = hashVal
    print("--- %s seconds ---" % (time.time() - loop_time))
    return sigMat


def getNeighbors(mat,user1):
    neighborTuples = set()
    neighborList = set()
    neighborList.add(user1)
    n, d = mat.shape
    for row in range(n):
        pairs = np.where(mat[row, :] == mat[row,user1])
        a1 = set(pairs[0])
        for user in a1:
            if user != user1:
                neighborTuples.add((user1,user))
                neighborList.add(user)
    return neighborTuples, neighborList


def getPairs(mat):
    ans = set()
    n, d = mat.shape
    bar = progressbar.ProgressBar()
    for row in bar(range(n)):
        x_original = mat[row, :]
        x = np.sort(x_original)
        indices = np.argsort(x_original)
        for ind in range(d):
            c = 1
            while ind + c < d and x[ind] == x[ind + c]:
                ind1 = indices[ind]
                ind2 = indices[ind + c]
                s_ind = min(ind1, ind2)
                b_ind = max(ind1, ind2)
                ans.add((s_ind, b_ind))
                c += 1
    return ans


def lsh(data,bands):
    r = data.shape[0]/bands
    largestPrime = prevprime(2**31-1)

    print "Num of Bands", bands, "r", r
    nusers = data.shape[1]
    # make new array
    sigMatA = np.zeros((bands,nusers),dtype=np.float64) # b x num of users
    newData = data.reshape(bands, r, nusers)
    a = np.random.randint(largestPrime, size=(bands, r, 1))
    a = np.repeat(a, nusers, axis=2)
    b = np.random.randint(largestPrime, size=(bands, r, 1))
    b = np.repeat(b, nusers, axis=2)
    new = (a*newData + b) % largestPrime
    sigMatA = np.sum(new, axis=(1))
    return sigMatA


def jaccard2(indexList,u1,u2):
    u1Movies = indexList[u1][0]
    u2Movies = indexList[u2][0]
    inter = np.intersect1d(u1Movies, u2Movies)
    union = np.union1d(u1Movies, u2Movies)
    try:
        return (float(np.size(inter)) / float(np.size(union)))
    except:
        return 0


def plot_hist(jsScores):
    hist, bins = np.histogram(jsScores, bins=25)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Jaccard Distance of Canidates')
    plt.savefig('JaccardHistFull.png')

def movie_id_correction(rating_data, movie_data):
    n = movie_data.shape[0]

    for row in range(n):
        id = movie_data[row]
        rating_data[np.where(rating_data[:,1]==id),1] = row + 1
    return rating_data

def readInAndConvert():
    start_time = time.time()

    print "Loading Dataset..."

    my_data = pd.read_csv('ml-20m/ratings.csv', delimiter=",")
    movie_data = pd.read_csv('ml-20m/movies.csv')
    movie_data['id'] = movie_data.index
    my_data = my_data.join(movie_data.set_index('movieId'), on = 'movieId', lsuffix = '_r' , rsuffix = '_m')
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    print "Converting to Numpy"
    my_data = my_data[['userId', 'id', 'rating']].as_matrix()
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    print "Converting Dataset to Desired Format..."
    mat = convert_matrix(my_data) # convert to required matrix, q1

    indexedList = getIndexes(mat)

    print("--- %s seconds to Convet & make Sparse---" % (time.time() - start_time))
    return indexedList,mat

def getIndexes(mat):
    indexedList = []
    bar = progressbar.ProgressBar()
    for col in bar(range(mat.shape[1])):
        indexedList.append(np.where(mat[:, col] == True))
    return indexedList

def sampleRandJaccard():
    userList, Ratings = readInAndConvert()
    outputList = []
    jsScores = np.zeros([10000, 1])
    bar = progressbar.ProgressBar()
    for index in range (10000):
        u1 =  random.randint(0, 138493 - 1)
        u2 =  random.randint(0, 138493 - 1)
        while u1 == u2:
            u2 = random.randint(0, 138493 - 1)
        score = jaccard2(userList,u1,u2)
        outputList.append([u1, u2, score])
        jsScores[index, 0] = score

    print "Mean Jaccard Similarity is ", np.mean(jsScores)

    hist, bins = np.histogram(jsScores, bins=25)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('Jaccard Distance of Canidates')
    plt.savefig('JaccardHist10000Pairs.png')

    outputList.sort(key=lambda x: x[2])

    outputList = outputList[-10:]
    csvfile = "canidates.csv"
    print outputList
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(outputList)


def runNeighbors(userList, user):
    total_time = time.time()
    newMat = userList
    userNumber = user
    numhash = [30]


    print len(numhash)
    for i in range(len(numhash)):
        nhash = numhash[i]
        print "Creating Signature Matrix by MinHash, ", nhash, " functions..."
        nusers = len(newMat)  # number of users
        nmovies = 27277
        sigMat = createSigMat(newMat,nusers,nmovies,nhash) # create initial signature matrix

        print "Looking for Canidate Pairs..."
        neighborTuples, neighborList = getNeighbors(sigMat,userNumber)


    print "Getting Jaccard Scores for ", len(neighborTuples), " pairs..."
    outputList = []
    jsScores = np.zeros([len(neighborTuples), 1])
    index = 0
    bar = progressbar.ProgressBar()
    for u1,u2 in bar(neighborTuples):
        score = jaccard2(userList,u1,u2)
        outputList.append([u1, u2, score])
        jsScores[index, 0] = score
        index += 1



    outputList.sort(key=lambda x: x[2])


    outputArray = np.array(outputList)

    highestScore = outputList[-1][2]

    outputArray = outputArray[np.where(outputArray[:,2] == highestScore)]
    for i in range(len(outputArray)):
        print "The Nearest Neighbor(s) to User ", user, "is User ", outputArray[i,1], "with a similarity score of ", outputArray[i,2]

    print("--- Total Runtime %s seconds ---" % (time.time() - total_time))


def main():

    userList,Ratings = readInAndConvert()


    numhash    = 800
    bandNumber = 100


    print "Creating Signature Matrix by MinHash, ", numhash, " functions..."
    nusers = len(userList)  # number of users
    nmovies = 27277
    sigMat = createSigMat(userList,nusers,nmovies,numhash) # create initial signature matrix

    print "LSH with..."

    sigMatA = lsh(sigMat,bandNumber) # create LSH of Sig Mat\

    print "Looking for Canidate Pairs..."

    canidates = getPairs(sigMatA)



    print "Getting Jaccard Scores for ", len(canidates), " pairs..."
    outputList = []
    jsScores = np.zeros([len(canidates), 1])
    index = 0
    bar = progressbar.ProgressBar()
    numPairs = 0
    for u1,u2 in bar(canidates):
        score = jaccard2(userList,u1,u2)
        if score >= 0.65:
            outputList.append([u1, u2, score])
            numPairs += 1
        jsScores[index, 0] = score
        index += 1

    plot_hist(jsScores)

    print "Found ", numPairs, " pairs with similarity over 0.65"

    csvfile = "canidates.csv"

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(outputList)



if __name__ == "__main__":
    total_time = time.time()
    # Q1.4
    main() ###one single hash

    # Q1.2
    #sampleRandJaccard() # q1.2 Sample Pairs.

    # Q1.5
    # userList, Ratings = readInAndConvert()
    # for i in range(100):
    #     runNeighbors(userList, random.randint(0,138492))




    print("--- Total Runtime %s seconds ---" % (time.time() - total_time))



import walker as w

import matplotlib.pyplot as plt
import numpy as np
import csv

flory_sample_size = 10000

def main():

    terramoto_activity()
    nAttempts_required(40)
    nAttempts_required(80)

def nAttempts_required(n):
    counter = 0
    validSAW = False
    while not validSAW:
        counter += 1
        walker = w.SmartSAW()
        validSAW = True
        for _ in range(n):
            validSAW = walker.walk() == 0
            if not validSAW:
                break

    return counter

def nAttempts_driver(iterations,n):
    res = []
    for i in xrange(iterations):
        if i % 10 == 0:
            print(str(round(100*(float(i)/iterations),2)) + r'% done')
        res.append(nAttempts_required(n))

    return np.mean(res)

def testing_flory():
    results = {}
    for i in xrange(flory_sample_size):
        if i % 100 == 0:
            print(str(round(100*(float(i)/flory_sample_size),2)) + r'% done')
        walker = w.SmartestSAW()
        validSAW = True
        while validSAW:
            validSAW = walker.walk() == 0
            if walker.nSteps > 500:
                break

        polymerization = str(walker.nSteps)
        R2 = walker.getR2()

        if polymerization in results.keys():
            results[polymerization].append(R2)
        else:
            results.update({polymerization  : [R2]})

    # this complex list comprehension translates the above results into a list
    # containing elements of the following format, sorted by increasing polymerization:
    # [polymerization, <R2>, nWalks]
    results_list = [[int(poly),np.mean(results[poly]),len(results[poly])] for poly in sorted(results.keys(), key=lambda t: int(t))]

    data2excel(results_list,'smartestSAW',multi_dim=True)


def terramoto_activity():
    # Activity code
    polymerization = np.array(range(1,10))
    R2_distribs = [w.allDistances(n) for n in range(1,10)]

    R_distribs = [[dist**0.5 for dist in distrib] for distrib in R2_distribs]

    mean_distances = np.array([np.mean(distrib) for distrib in R_distribs])
    nWalks = np.array([len(distrib) for distrib in R2_distribs])

    data2excel(mean_distances, 'mean_distances_1-9_R')

    plt.scatter(polymerization,mean_distances)
    plt.show()
    plt.gcf().clear()
    plt.scatter(polymerization,nWalks)
    plt.show()

    total_number_of_walks = sum(nWalks)
    walks_studied_per_year = 1 * 60 * 8 * 5 * 52    # per minute * hour * day * week * year

    years_spent = round(float(total_number_of_walks) / walks_studied_per_year, 2)

    print('At %d walks, Teramoto spent %.2f years' % (total_number_of_walks, years_spent))

def data2excel(data,fname,multi_dim=False):
    with open(fname + '.csv', 'wb') as file:
        if not multi_dim:
            wr = csv.writer(file,delimiter=' ')
            for elem in data:
                wr.writerow([elem])
        else:
            wr = csv.writer(file,delimiter=',')
            wr.writerows(data)

if __name__ == "__main__":
    main()

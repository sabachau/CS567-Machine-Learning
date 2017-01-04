import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import itertools
import random
from scipy.stats import multivariate_normal

def calc_cost(mean_arr,cluster_dctnry):
    sum = 0
    total_pts = 0
    for cluster_num in cluster_dctnry:
        points = cluster_dctnry[cluster_num]
        total_pts += len(points)
        mean_of_cluster = mean_arr[cluster_num] # np.mean(cluster_dctnry[cluster_num],axis=0)
        for point in points:
            diff = point-mean_of_cluster
            sq_diff = np.power(diff,2)
            sum += np.sum(sq_diff, axis=0)
    cost_J = sum/total_pts
    return cost_J

def plot_for_five(cluster_dctnry):
    print 'Displaying plot'
    circle = genfromtxt('hw5_circle.csv', delimiter=',')
    colors = itertools.cycle(["r", "b", "g","c","m","y","k"])
    for key in cluster_dctnry:
        iters_likelihood = cluster_dctnry[key]
        points_x = np.array(range(len(iters_likelihood)))
        points_y = np.array(iters_likelihood)
        plt.scatter(points_x, points_y, color=next(colors), alpha=0.5)
    plt.show()
    return

def plot(cluster_dctnry):
    print 'Displaying plot'
    circle = genfromtxt('hw5_circle.csv', delimiter=',')

    colors = itertools.cycle(["r", "b", "g","c","m","y","k"])
    for key in cluster_dctnry:
        # print 'k',key
        if cluster_dctnry[key].any():
            points = cluster_dctnry[key]
            points_x = points[:,[0]]
            points_y = points[:,[1]]
            plt.scatter(points_x, points_y, color=next(colors), alpha=0.5)
    plt.show()
    print 'Resuming execution'
    return

def transform_dataset(dataset):

    tr_dataset = np.square(dataset[:,[0]])+np.square(dataset[:,[1]])
    return tr_dataset

def convertindicestopoint(dataset,cluster_allot):
    dct = {}
    for k in cluster_allot:
        if k not in dct:
            dct[k]=[]
        indices = cluster_allot[k]
        for idx in indices:
            dct[k].append(dataset[idx])
        dct[k] = np.asarray(dct[k])
    return dct

def kernel_kmeans(dataset,K):
    orig_data = dataset
    dataset=transform_dataset(dataset)
    gamma = 20 # gamma = 1/sigma^2
    generate_kernel_matrix(dataset, gamma)
    list_indices = range(dataset.shape[0])
    list_indices_copy = list_indices
    cluster_allot={}
    num = len(list_indices)
    rem = random.randint(0, num)  # between 0 and num
    pt = []
    for k in range(K):
        if k == K - 1:
            pt.append(num)
            break
        if rem > 0:
            nums = rem
            pt.append(nums)
            rem = random.randint(0, num - nums)
            num = num - nums
        else:
            pt.append(0)
    for k in range(K):
        cluster_allot[k] = random.sample(list_indices_copy, pt[k])
        list_indices_copy = [x for x in list_indices_copy if x not in cluster_allot[k]]
    kernel_matrix = genfromtxt('kernel_matrix.csv', delimiter=',')
    count = 0
    while True:
        count+=1
        print 'Iteration: ',count
        cluster=cluster_allot
        cluster_allot={}
        for k in range(K):
            if k not in cluster_allot:
                cluster_allot[k]=[]
        for idx,point in enumerate(dataset):
            term1 = kernel_matrix[idx][idx]
            distance_from={}
            for k in range(K):
                term3=0
                num_points=len(cluster[k])
                for a_j in cluster[k]:
                    for a_l in cluster[k]:
                        term3 += kernel_matrix[a_j][a_l]
                term3/=(num_points**2)
                term2 = 0
                for point_k in cluster[k]:
                    term2+=kernel_matrix[idx][point_k]
                term2 *= (-2)
                term2 /= num_points
                distance_from[k] = term1+term2+term3
            min_cluster_num = min(distance_from, key=distance_from.get)
            cluster_allot[min_cluster_num].append(idx)
        if np.array_equal(cluster,cluster_allot):
            break
    cluster_allot = convertindicestopoint(orig_data,cluster_allot)
    plot(cluster_allot)#,min_cluster_num)

def kmeans_algo(dataset):
    K = [ 2,3,5]
    cost = {}
    point_distribution = {}
    for k in K:
        '''
        To avoid the problem of getting stuck in a local minima, executed k-means
        with multiple random initializations for each k and took the final error for that cluster to be
        the smallest of error obtained from multiple initializations
        '''
        for i in range(100):
            mean_k_d = dataset[np.random.choice(dataset.shape[0], k, replace=False), :]
            new_mean = np.asarray(mean_k_d - 1)
            run = 1
            while True:
                cluster = {}
                if run == 1:
                    new_mean = mean_k_d
                    run = 2
                mean_k_d = new_mean
                for point in dataset:
                    diff = []
                    diff.append(np.sum(np.square(point - mean_k_d), axis=1))
                    cluster_num = np.nanargmin(diff, axis=1)
                    if cluster_num[0] not in cluster:
                        cluster[cluster_num[0]] = []
                    cluster[cluster_num[0]].append(point)
                mean = []
                for key in cluster:
                    cluster[key] = np.asarray(cluster[key])
                    if cluster[key].any():
                        mean.append(np.mean(cluster[key], axis=0))
                    else:
                        mean.append(np.zeros((1, 2)))
                new_mean = np.asarray(mean)
                if np.array_equal(mean_k_d, new_mean):
                    c_final = cluster
                    mean_final = new_mean
                    break
            cost_k = calc_cost(mean_final, c_final)
            if k not in cost:
                cost[k] = float("inf")
            if cost_k < cost[k]:
                cost[k] = cost_k
                point_distribution = c_final
        print 'For k=',k,'min error was found to be: ',cost[k]
        plot(point_distribution)
    print 'Cost List:',cost
    min_cluster_num = min(cost, key=cost.get)
    print 'best K: ', min_cluster_num

def generate_kernel_matrix(dataset,gamma):
    final_kernel=[]
    for idx,point in enumerate(dataset):
        k=[]
        for idx2,point2 in enumerate(dataset):
            a = np.square(point - point2)
            b = np.sum(a, axis = 0)
            k.append(np.exp(-gamma*b)) #taking sigma=1
        final_kernel.append(k)
    final_kernel = np.asarray(final_kernel)
    np.savetxt('kernel_matrix.csv', final_kernel, delimiter=',')

def EM_algorithm(dataset):
    num_clusters = 3
    cluster_allot = {}
    num = dataset.shape[0]
    list_indices = range(num)
    list_indices_copy = list_indices
    list_rndm_nos = random.randint(0, num) #between 0 and num
    pt = []
    for k in range(num_clusters):
        if k==num_clusters-1:
            pt.append(num)
            break
        if list_rndm_nos > 0:
            updates = list_rndm_nos
            pt.append(updates)
            list_rndm_nos = random.randint(0, num - updates)
            num=num-updates
        else:
            pt.append(0)
    m = dataset.shape[0]
    for k in range(num_clusters):
        cluster_allot[k] = random.sample(list_indices_copy, pt[k])
        list_indices_copy = [x for x in list_indices_copy if x not in cluster_allot[k]]
    pst_prob_cx = [[0 for x in range(dataset.shape[0])] for y in range(num_clusters)]
    total_pst_prob=[0 for x in range(num_clusters)]
    prob_c=[0 for x in range(num_clusters)]
    for k in range(num_clusters):
        indxs = cluster_allot[k]
        for idx in indxs:
            if idx in cluster_allot[k]:
                pst_prob_cx[k][idx] = 1
                prob_c[k]+=pst_prob_cx[k][idx]
                total_pst_prob[k]+=1
        prob_c[k]=prob_c[k]/float(m)
    prior = []
    for c in total_pst_prob:
        prior.append(float(c)/m)
    mean_of_cluster = []
    chunks=[]
    for k,arr in enumerate(pst_prob_cx):
        chunked = np.array([np.tile([np.array(x)],dataset.shape[1]) for x in arr])
        chunks.append(chunked)
        mean_of_cluster.append(np.sum(dataset*chunked,axis=0)/total_pst_prob[k])
    cov_matrix = {}
    for k in range(num_clusters):
        cov = []
        for d1 in range(dataset.shape[1]):
            cv = []
            for d2 in range(dataset.shape[1]):
                temp = dataset-mean_of_cluster[k]
                temp2 = temp[:,[d1]]*temp[:,[d2]]
                tr_post_pro = np.array([np.array([float(e)/(total_pst_prob[k]-1)]) for e in pst_prob_cx[k]])
                temp3 = np.sum(tr_post_pro*temp2)
                cv.append(temp3)
            cov.append(cv)
        cov_matrix[k] = np.asarray(cov)
    likelihood_arr=[]
    new_l = 5000000
    iterations = 0
    while(True):
        old_l = new_l
        pst_prob_cx = expectation_step(prob_c, dataset, mean_of_cluster, cov_matrix, num_clusters)
        prob_c, mean_of_cluster, cov_matrix,total_pst_prob = maximization_step(pst_prob_cx, dataset, num_clusters)
        new_l = calc_likelihood(prob_c,dataset,mean_of_cluster,cov_matrix,num_clusters)
        likelihood_arr.append(new_l)
        print 'iteration: ',iterations,' log-likelihood: ',new_l
        if(abs(new_l-old_l<0.001) and iterations>4):
            break
        iterations+=1
    cluster_allot ={}
    for i in range(dataset.shape[0]):
        min_prob = float("-inf")
        for k in range(num_clusters):
            if k not in cluster_allot:
                cluster_allot[k]=[]
            prob = pst_prob_cx[k][i]
            if(prob>min_prob):
                min_prob = prob
                min_idx = k
        cluster_allot[min_idx].append(i)
    plot(convertindicestopoint(dataset,cluster_allot))
    return likelihood_arr,mean_of_cluster,cov_matrix

def expectation_step(all_p_sum, dataset, mean_of_clusters, cov_matrix, num_clusters):
    post_prob_xc = [[0 for i in range(dataset.shape[0])] for j in range(num_clusters)]
    nr = [[0 for i in range(dataset.shape[0])] for j in range(num_clusters)]
    dr=[0 for c in range(dataset.shape[0])]
    for idx,point in enumerate(dataset):
        for k in range(num_clusters):
            nr[k][idx] = all_p_sum[k] * multivariate_normal.pdf(point, mean=mean_of_clusters[k], cov=cov_matrix[k])
            dr[idx]+= nr[k][idx]
    for k in range(num_clusters):
        for idx, point in enumerate(dataset):
            post_prob_xc[k][idx] =float (nr[k][idx])/dr[idx]
    return post_prob_xc

def maximization_step(tr_post_prob, dataset, num_clusters):
    all_sum=[]
    total_post_prob=[0 for i in range(num_clusters)]
    mean_of_cluster = [0 for i in range(num_clusters)]
    for k in range(num_clusters):
        lst = tr_post_prob[k]
        total_post_prob[k]=sum(lst)
        mean_of_cluster[k] = np.average(dataset, axis=0, weights=lst)
    n = sum(total_post_prob)
    for k in range(num_clusters):
        all_sum.append(float(total_post_prob[k])/n)
    cov_matrix = calc_cov(num_clusters, dataset, mean_of_cluster, all_sum, tr_post_prob, total_post_prob)
    return all_sum,mean_of_cluster,cov_matrix,total_post_prob

def calc_likelihood(pi_c, dataset, mean_of_cluster, cov_matrix, num_clusters):
    likelihood = 0
    for point in dataset:
        cluster_wise=0
        for k in range(num_clusters):
            cluster_wise += pi_c[k]*multivariate_normal.pdf(point, mean=mean_of_cluster[k], cov=cov_matrix[k])
        likelihood += np.log(cluster_wise)
    return likelihood

def calc_cov(num_clusters, dataset, mean_of_cluster, pi_c, tr_post_prob, total_pst_prob):
    cov_matrix = {}
    for k in range(num_clusters):
        cov = []
        for d1 in range(dataset.shape[1]):
            cv = []
            for d2 in range(dataset.shape[1]):
                temp = dataset - mean_of_cluster[k]
                temp2 = temp[:, [d1]] * temp[:, [d2]]
                tr_post_prob = np.array([np.array([float(e) / (total_pst_prob[k] - 1)]) for e in tr_post_prob[k]])
                temp3 = np.sum(tr_post_prob * temp2)
                cv.append(temp3)
            cov.append(cv)
        cov_matrix[k] = np.asarray(cov)
    return cov_matrix

def main():
    blob = genfromtxt('hw5_blob.csv', delimiter=',')
    circle = genfromtxt('hw5_circle.csv', delimiter=',')
    print '|-----KMeans---|'
    print '___________|-----Blob dataset-----|_________'
    kmeans_algo(blob)
    print '___________|--------|_________'
    kmeans_algo(circle)
    print '\n\n|-----Kernel KMeans, K=2, Circle dataset---|'
    kernel_kmeans(circle,K = 2)
    print '\n\n|-----EM Algorithm, K=3, Blob dataset---|'
    five_runs={}
    for run in range(5):
        print 'run: ',run
        if run not in five_runs:
            five_runs[run] = []
        five_runs[run],mean_of_cluster,cov_matrix= EM_algorithm(blob)
        print 'Mean: ',mean_of_cluster
        print 'Covariance Matrix', cov_matrix
    plot_for_five(five_runs)

if __name__=='__main__':
    main()
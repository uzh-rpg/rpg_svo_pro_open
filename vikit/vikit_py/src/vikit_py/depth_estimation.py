#!/usr/bin/python3

import numpy as np
from scipy.stats import beta
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Cardo']})
rc('text', usetex=True)

def q_of_Z_pi(Z, RHO, mu_sigma2_a_b):
    q = norm.pdf(Z, mu_sigma2_a_b[0], np.sqrt(mu_sigma2_a_b[1])) * beta.pdf(RHO, mu_sigma2_a_b[2], mu_sigma2_a_b[3]); 
    return q
    
def update_filter_vogiatzis(z, tau2, mu_range, mu_sigma2_a_b):
    mu = mu_sigma2_a_b[0]
    sigma2 = mu_sigma2_a_b[1]
    a = mu_sigma2_a_b[2]
    b = mu_sigma2_a_b[3]

    norm_scale = np.sqrt(sigma2 + tau2)
    
    s2 = 1.0/(1.0/sigma2 + 1.0/tau2)
    m = s2*(mu/sigma2 + z/tau2)
    uniform_x = 1.0/mu_range
    C1 = a/(a+b) * norm.pdf(z, mu, norm_scale)
    C2 = b/(a+b) * uniform_x
    normalization_constant = C1 + C2
    C1 /= normalization_constant
    C2 /= normalization_constant
    f = C1*(a+1.0)/(a+b+1.0) + C2*a/(a+b+1.0)
    e = C1*(a+1.0)*(a+2.0)/((a+b+1.0)*(a+b+2.0)) \
      + C2*a*(a+1.0)/((a+b+1.0)*(a+b+2.0));

    # update parameters
    mu_new = C1*m+C2*mu
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new
    mu = mu_new
    a = (e - f) / (f - e / f)
    b = a * (1.0 - f) / f
    
    mu_sigma2_a_b = np.array([mu, sigma2, a, b])
    return mu_sigma2_a_b

def update_filter_gaussian(z, tau2, mu_sigma2):
    mu = mu_sigma2[0]
    sigma2 = mu_sigma2[1]

    denom = (sigma2 + tau2)
    mu = (sigma2 * z + tau2 * mu) / denom
    sigma2 = sigma2 * tau2 / denom
    mu_sigma2 = np.array([mu, sigma2])
    return mu_sigma2

def plot_distribution_vogiatzis(ax, mu_sigma2_a_b, Z, RHO):
    Q = q_of_Z_pi(Z, RHO, mu_sigma2_a_b)
    ax.imshow(Q, aspect='auto', origin='lower')
    
def plot_histogram(ax, z_meas_vec, z_min, z_max):
    bins = np.linspace(z_min, z_max, 40)
    ax.hist(z_meas_vec, bins, color='blue', edgecolor='w') # none
    ax.set_xlim([z_min, z_max])
    
def plot_gaussian(ax, mu_sigma2, z_min, z_max):
    x = np.linspace(z_min, z_max, 1000)
    y = norm.pdf(x, mu_sigma2[0], np.sqrt(mu_sigma2[1]))
    ax.plot(x, y, lw=2.5, color='b')
    ax.set_xlim([z_min, z_max])
    
def experiment_1(inlier_prob = 0.7):
    # init state
    z_max = 1.0/0.5
    z_min = 0.0
    z_true = 1.0/4.0
    z_init = 1.0/1.0
    tau = 0.1
    a_b_init = 1.5
    N = 30              # num updates
    mu_range = z_max - z_min
    sigma2_init = mu_range * mu_range / 36.0
    tau2 = tau*tau
    mu_sigma2_a_b = [z_init, sigma2_init, a_b_init, a_b_init] # for vogiatzis
    mu_sigma2 = [z_init, sigma2_init]                         # for gaussian
    
    
    # prepare plotting
    depth_samples = np.linspace(z_min, z_max, 400)
    inlier_ratios = np.linspace(0, 1, 100)
    Z, RHO = np.meshgrid(depth_samples, inlier_ratios)
    axes_vec = list()    
    fig1, axes = plt.subplots(3, 1, figsize=(5, 3))
    axes_vec.append(axes)
    fig2, axes = plt.subplots(3, 1, figsize=(5, 3))
    axes_vec.append(axes)
    fig3, axes = plt.subplots(3, 1, figsize=(5, 3))
    axes_vec.append(axes)
    
    z_meas_vec = list()
    
    # perform measurement updates
    for i in range(N):
        
        # measurement is inlier with certain probability
        t = np.random.choice(['inlier', 'outlier'], p=[inlier_prob, (1.0-inlier_prob)])
        if t == 'inlier':
            z_meas = z_true  +  np.random.normal(0.0, tau)
        elif t == 'outlier':
            z_meas = np.random.uniform(z_min, z_max)
        z_meas_vec.append(z_meas)
            
        mu_sigma2_a_b = update_filter_vogiatzis(z_meas, tau2, mu_range, mu_sigma2_a_b)
        mu_sigma2 = update_filter_gaussian(z_meas, tau2, mu_sigma2)
        if(i == 2):
            plot_distribution_vogiatzis(axes_vec[0][2], mu_sigma2_a_b, Z, RHO)
            plot_gaussian(axes_vec[0][1], mu_sigma2, z_min, z_max)
            plot_histogram(axes_vec[0][0], z_meas_vec, z_min, z_max)
        if(i == N/2):
            plot_distribution_vogiatzis(axes_vec[1][2], mu_sigma2_a_b, Z, RHO)
            plot_gaussian(axes_vec[1][1], mu_sigma2, z_min, z_max)
            plot_histogram(axes_vec[1][0], z_meas_vec, z_min, z_max)
        if(i == N-1) :
            plot_distribution_vogiatzis(axes_vec[2][2], mu_sigma2_a_b, Z, RHO)
            plot_gaussian(axes_vec[2][1], mu_sigma2, z_min, z_max)
            plot_histogram(axes_vec[2][0], z_meas_vec, z_min, z_max)
    
    def format_axes(axes, n_iter):
        
        for i in range(3):
            ax = axes[i]

            ax.xaxis.set_ticks_position('none') # don't display tick marks
            ax.yaxis.set_ticks_position('none')
    
            if i%3 == 2: # vogiatzis:
              #ax.set_title('After '+str(int(n_iter))+' measurements, inlier probability = '+str(inlier_prob))
              ax.set_ylabel(r"$\gamma$", rotation=0)
              ax.set_yticks([0,50,100])
              ax.set_yticklabels([0,0.5,1])
              ax.set_xlabel('Inverse Depth')
            if i%3 == 1: # gaussian:
              #ax.locator_params(axis='y', nbins=3)
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
              ax.set_yticks([])
              ax.set_xticks([])
              ax.tick_params('x',top='off')
            if i%3 == 0: # histogram
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
              ax.set_yticks([])
              ax.set_xticks([])
              
            #width = 1.0 - 2*h_space
            #height = (1.0 - v_space) / n_plots - v_space
            #ax.set_position([h_space, 1-(i+1)*(height+v_space)+v_space/2, width, height])
        
    
    format_axes(axes_vec[0], 3)
    format_axes(axes_vec[1], N/2)
    format_axes(axes_vec[2], N)
    
    fig1.tight_layout()
    fig1.savefig('depth_estimation_1.pdf')
    fig2.tight_layout()
    fig2.savefig('depth_estimation_2.pdf')
    fig3.tight_layout()
    fig3.savefig('depth_estimation_3.pdf')
    
    print('error vogiatzis = ' + str(np.abs(mu_sigma2_a_b[0]-z_true)))
    print('error gaussian = ' + str(np.abs(mu_sigma2[0]-z_true)))
    
    
experiment_1(0.7)
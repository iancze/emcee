import numpy as np
import emcee

#Designed to test the line example with emcee

from gibbs import line, load_data, plot_triangle, sigma

xs, ys = load_data() #loads xs and ys from file

def lnprob(p):
    b, m = p
    ys_model = line(xs, b, m)
    chi2 = np.sum((ys_model - ys)**2)/sigma**2
    return -0.5 * chi2

def main():
    ndim = 2
    nwalkers = 50
    p0 = np.array([np.random.uniform(0.5, 2.0, size=(nwalkers,)), np.random.uniform(1.5, 3.5, size=(nwalkers,))]).T
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    pos, prob, state = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    sampler.run_mcmc(pos, 8000, rstate0=state)

    #plot_data()
    samples = sampler.flatchain
    print(samples)
    plot_triangle(samples, fname="triangle_emcee.png")
    pass

if __name__=="__main__":
    main()

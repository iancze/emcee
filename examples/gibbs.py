#hacked Metropolis-Hastings sampler

from emcee import GibbsSampler, GibbsController
import numpy as np
import matplotlib.pyplot as plt

def line(x, b, m):
    '''
    Generate a line with intercept b and slope m, using the series of x points
    '''
    return b + m * x

class Line:
    '''
    State-full model of a line, designed to mimic the more advanced StellarSpectra model.
    '''
    def __init__(self, xs, ys, sigma, debug=False):
        self.xs = xs
        self.ys = ys
        self.sigma = sigma
        self.debug = debug
        self.ys_model = None
        self.ys_old = None
        self.b = None
        self.m = None

    def update_b(self, b):
        self.b = b

    def update_m(self, m):
        self.m = m

    def downsample(self):
        if self.debug:
            print("Downsampling to new model")
        self.old_ys = self.ys_model
        self.ys_model = line(self.xs, self.b, self.m)

    def revert(self):
        if self.debug:
            print("Reverting back to old model")
        self.ys_model = self.old_ys

    def evaluate(self):
        chi2 = np.sum((self.ys_model - self.ys)**2)/self.sigma**2
        return -0.5 * chi2

    def lnprob_b(self, p):
        b = p[0]
        if self.debug:
            print("calling lnprob_b with b={}".format(b))
        self.update_b(b)
        self.downsample()
        return self.evaluate()

    def lnprob_m(self, p):
        m = p[0]
        if self.debug:
            print("calling lnprob_m with m={}".format(m))
        self.update_m(m)
        self.downsample()
        return self.evaluate()

b = 1.2
m = 2.5
sigma = 2.

def generate_data():
    xs = np.sort(np.random.uniform(low=0, high=10, size=(20,)))
    noise = np.random.normal(scale=sigma, size=(len(xs),))
    ys = line(xs, b, m) + noise
    np.save("xs.npy", xs)
    np.save("ys.npy", ys)

def load_data():
    global xs,ys
    xs = np.load("xs.npy")
    ys = np.load("ys.npy")
    return xs, ys

load_data()

def plot_data():
    plt.errorbar(xs, ys, yerr=sigma, fmt="o")

    plt.plot(xs, line(xs, b, m))
    plt.title(r"$b={:.2f}$, $m={:.2f}$".format(b, m))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig("data.png")


#MHCov = np.array([0.25, 0.25])**2 * np.identity(2)


def plot_samples(samples):
    '''
    Plot the chain value as a function of position
    '''
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(samples[:,0])
    ax[0].set_xlabel(r"$b$")
    ax[1].plot(samples[:,1])
    ax[1].set_xlabel(r"$m$")
    fig.savefig("samples.png")

def plot_triangle(samples, fname="triangle.png"):
    '''
    Use triangle.py to plot the samples.
    '''
    import triangle
    figure = triangle.corner(samples, labels=(r"$b$", r"$m$"), quantiles=[0.16, 0.5, 0.84], truths=[b, m], 
            show_titles=True, title_args={"fontsize": 12})
    figure.savefig(fname)



def main():
    #plot_data()

    model = Line(xs, ys, sigma, debug=False)
    model.update_b(1.1)
    model.update_m(2.2)
    model.downsample()

    sampler_b = GibbsSampler(np.array([0.5])**2, np.array([1.1]), model.revert, 1, model.lnprob_b)
    sampler_m = GibbsSampler(np.array([0.4])**2, np.array([2.2]), model.revert, 1, model.lnprob_m)

    sampler = GibbsController([sampler_b, sampler_m], debug=False)
    sampler.run(10000)
    sampler.reset()
    sampler.run(200000)

    print(sampler.acceptance_fraction())
    print(sampler.acor())

    samples = sampler.flatchain()
    print(samples)
    plot_samples(samples)
    plot_triangle(samples)
    pass

if __name__=="__main__":
    main()

#Alternate sampling strategy with EnsembleSampler


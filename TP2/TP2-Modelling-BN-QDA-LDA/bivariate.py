import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BivariateDensity:
    def __init__(self, X, Y, density, variables=['X','Y'], name=None):
        self.X, self.Y = X, Y
        self.density = density
        self.variables = variables
        if name is None:
            name = f'p({variables[0]}, {variables[1]})'
        self.name = name
        
    def plot(self, colors=None, n_levels=10):
        ax = plt.gca()
        g = plt.contour(self.X, self.Y, self.density, n_levels, colors=colors)
        ax.clabel(g, inline=True, fontsize=10)
        plt.xlabel(self.variables[0])
        plt.ylabel(self.variables[1])
        plt.title(self.name)

    def conditional_density(self, conditioned_var, threshold=0.1):
        cond_dim = self.variables.index(conditioned_var)
        if cond_dim == 1:
            density = self.density
            X,Y = self.X, self.Y
            variables = self.variables
        else:
            density = self.density.T
            X,Y = self.Y, self.X
            variables = list(reversed(self.variables))
        marginal = np.sum(density, 1)        
        max_marginal = np.max(marginal)
        I = (marginal > max_marginal * threshold)
        density, marginal, Y = density[I,:], marginal[I], Y[I]
        density = density / marginal.reshape((-1,1))
        return BivariateDensity(X, Y, density, variables, f'p({variables[0]} | {variables[1]})')

def get_bivariate_density(df, var1, var2, kde_factor=1., nbr_points=100, margin_factor=0.1):
        df2 = df[[var1,var2]].dropna()
        mins, maxs = df2.min(), df2.max()
        margins = margin_factor *(maxs - mins)
        mins -= margins
        maxs += margins
        
        kernel = stats.gaussian_kde(df2.to_numpy().T, bw_method='scott')
        kernel.set_bandwidth(kernel.factor * kde_factor)

        X = np.linspace(mins[var1], maxs[var1], nbr_points)
        Y = np.linspace(mins[var2], maxs[var2], nbr_points)
        XX, YY = np.meshgrid(X,Y)
        XY = np.vstack([XX.ravel(), YY.ravel()])
        densities = kernel(XY).T.reshape((len(Y), len(X)))
        return BivariateDensity(X, Y, densities, [var1, var2])
    

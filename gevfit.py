import scipy.stats as stats
import scipy.special as special
import numpy as np
from sklearn.utils import resample

# We also want to check information on the maxima, which is harder since GEV estimation is tricky.
# We use an alternative method for this estimation from XXX to attempt this that Chris Leap implemented
def _pwm_moment(xs, moment):
        samples = np.sort(xs)
        n = samples.size
        res = 0
        for j, sample in enumerate(samples):
                term = sample
                for i in range(1,moment + 1):
                        term = term * (j + 1 - i) / (n - i)
                res = res + term
        return res / n

def _pwm_fit(samples):
	# Generate moments from sample data
	samples = np.sort(samples)
	n = samples.size
	b0 = _pwm_moment(samples,0)
	b1 = _pwm_moment(samples,1)
	b2 = _pwm_moment(samples,2)

	# Generate GEV parameters
	c = (2*b1-b0)/(3*b2-b0) - np.log(2)/np.log(3)
	est_shape = 7.8590*c+2.9554*np.power(c,2)
	gamma = special.gamma(1 + est_shape)
	est_scale = est_shape*(2*b1-b0)/(gamma*(1-np.power(2,-est_shape)))
	est_loc = b0 + est_scale*(gamma-1)/est_shape
	return est_shape,est_loc,est_scale

def fit(samples, method='pwm'):
	if (method == 'mle'):
		return stats.genextreme.fit(samples, 0)
	elif (method == 'pwm'):
		return _pwm_fit(samples)
	else:
		print("Unknown fit method {1}".format(method))
		return None

def fit_ci(samples, bootstraps=1000, fraction=1.0, method='pwm', alpha=0.95, variant='nonparametric percentile'):
	# First get the primary fit from the samples themselves
	primary_fit = fit(samples, method=method)

	values = [[], [], []]
	for _ in range(0, bootstraps):
		# Get a bootstrap sample
		if (variant == 'nonparametric percentile'):
			bootstrap = resample(samples)
		elif (variant == 'parametric percentile'):
			bootstrap = stats.genextreme.rvs(primary_fit[0], loc=primary_fit[1], scale=primary_fit[2],
						         size=len(samples))
		else:
			print('Unknown confidence interval determination variant')
			return None

		sample_fit = fit(bootstrap, method=method)
		for i in range(0,3):
			values[i].append(sample_fit[i])

	ci=[(), (), ()]
	for i in range(0,3):
		p = ((1.0-alpha)/2.0) * 100
		lower = np.percentile(values[i], p)
		p = (alpha+((1.0-alpha)/2.0)) * 100
		upper = np.percentile(values[i], p)
		ci[i] = (lower, upper)

	return primary_fit, ci
	
if (__name__ == '__main__'):
	shape=0.0
	loc=1000
	scale=100
	print("Generating GEV data with shape: {}, loc: {}, scale: {}".format(shape, loc, scale))
	samples = stats.genextreme.rvs(c=shape, loc=loc, scale=scale, size=1000)
	f = fit_ci(samples, method='pwm', variant='nonparametric percentile')
	print("Fit parameters: {}".format(f[0]))
	print("Fit confidence intervals: {}".format(f[1]))
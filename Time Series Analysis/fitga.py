import numpy as np
import scipy
import scipy.special
gamma = scipy.special.gamma

# this is a function to make a GARCH(1,1) timeseries of length N 

def generateX(N, omega, alpha ,beta, nu, sigma1):

	X = np.zeros(N)
	sigmasquared = sigma1 * np.ones(N)
	Z = np.sqrt((nu - 2) / nu) * np.random.standard_t(nu, N)

        X[1] = np.sqrt(sigmasquared[1]) * Z[1]
        
	for i in range(2, N):
		sigmasquared[i] = omega + alpha * sigmasquared[i - 1] + beta * X[i - 1] ** 2
		X[i] = np.sqrt(sigmasquared[i]) * Z[i]

	return X




# this is a function that given a timeseries X and a set of parameters params finds MINUS the log likelihood
# following GARCH 

def logli(params, X):
	N = len(X)
        omega, alpha, beta, nu, sigma1 = params 
	sigmasquared = sigma1 ** 2 * np.ones(len(X))

	for i in range(2, N):
		sigmasquared[i] = omega + alpha * sigmasquared[i - 1] + beta * X[i - 1] ** 2

	return - ( N * np.log( gamma((nu + 1)/2) / gamma(nu/2) / np.sqrt((nu - 2)* np.pi) )    \
		- np.sum( (nu + 1)/2 * np.log(1 + np.divide(np.power(X, 2), sigmasquared)/(nu - 2)) \
		          + np.log(sigmasquared)/2 ) )

















# this function generates GARCH(1,1) data for a certain set of parameters
# and then fits parameters to the data 
# it repeats this "runs" time ....  and then shows the results 

def chkft(runs):

	N = 10000
	omega = 0.1
	alpha = 0.5
	beta = 0.4
	nu = 4.0
	sigma1 = omega / (1 - alpha - beta)
	params = np.array([omega, alpha, beta, nu, sigma1])
	
	myd = np.zeros((runs, len(params)))

	for i in range(runs):
		x = generateX(N, *params)
		myd[i, :] = scipy.optimize.fmin(logli, x0 = params, args = (x, ), xtol = 1e-5, ftol = 1e-10)

        print myd



        
# fitting to some actual data: dat1  TASE returns from 2000-2017 (maybe backwards)
#                              dat2  IBM daily returns from 2010-2012 (maybe backwards)
        

x1 = np.array(file('dat1').read().splitlines()).astype(np.float)
x2 = np.flip(x1,0)
y1 = np.array(file('dat2').read().splitlines()).astype(np.float)
y2 = np.flip(y1,0) 
sw = np.array([0.1, 0.5, 0.4, 5.1, 1.5])

a1 = scipy.optimize.fmin(logli, x0 = sw, args = (x1, ), xtol = 1e-5, ftol = 1e-10)
a2 = scipy.optimize.fmin(logli, x0 = sw, args = (x2, ), xtol = 1e-5, ftol = 1e-10)
b1 = scipy.optimize.fmin(logli, x0 = sw, args = (y1, ), xtol = 1e-5, ftol = 1e-10)
b2 = scipy.optimize.fmin(logli, x0 = sw, args = (y2, ), xtol = 1e-5, ftol = 1e-10)
        

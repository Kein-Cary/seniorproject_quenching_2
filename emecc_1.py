import numpy as np
import matplotlib.pyplot as plt
# choose the "True" parameter
m_true = -0.9594
b_true = 4.294
f_true = 0.534
# Generate some synthetic data from the model
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y_true = m_true*x+b_true
y += np.abs(f_true*y)*np.random.randn(N)
y += yerr*np.random.randn(N)
##first result
plt.plot(x,y_true,'r-')
plt.errorbar(x,y,yerr= yerr,fmt='g*')
plt.show()
A = np.vstack((np.ones_like(x),x)).T
C = np.diag(yerr*yerr)
cov = np.linalg.inv(np.dot(A.T,np.linalg.solve(C,A)))
b_ls, m_ls = np.dot(cov,np.dot(A.T, np.linalg.solve(C,y)))
print('m=',m_ls)
print('b=',b_ls)
y_ls = m_ls*x+b_ls
##second result
plt.plot(x,y_true,'r-')
plt.plot(x,y_ls,'b--')
plt.errorbar(x,y,yerr= yerr,fmt='g*')
plt.show()
def lnlike(theta, x, y, yerr):
	m, b, lnf = theta
	model = m*x+b
	inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

import scipy.optimize as op
nll = lambda * args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]
print('m_like=',m_ml)
print('b_like=',b_ml)
print('f_like=',lnf_ml)
y_ml= m_ml*x+b_ml
##third result
plt.plot(x,y_true,'r-')
plt.plot(x,y_ls,'b--')
plt.plot(x,y_ml,'m--')
plt.errorbar(x,y,yerr= yerr,fmt='g*')
plt.show()
def lnprior(theta):
	m, b, lnf = theta
	if -5.0<m<0.5 and 0.0<b<10.0 and -10.0<lnf<1.0:
		return 0.0
	return -np.inf 

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
	   return -np.inf
	return lp + lnlike(theta, x, y, yerr)

ndim, nwalkers = 3, 100
pos = [result["x"]+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
##next is The Monto Caro 
import emcee  
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 50:, :].reshape((-1,ndim))
##result
import corner
fig = corner.corner(samples, labels = ["$m$", "$b$", "$\ln\,f$"],\
	truths = [m_true, b_true, np.log(f_true)])
#fig.savefig('practice',dpi=600)
plt.show()
xl = np.array([0, 10])
for m, b, lnf in samples[np.random.randint(len(samples),size=100)]:
	plt.plot(xl, m*xl+b,'k',alpha=0.2)
plt.plot(xl, m_true*xl+b_true,'r',lw=2, alpha=0.8)
plt.errorbar(x, y, yerr=yerr, fmt = ".k")
plt.show()
# samples[:,2] = np.exp(samples[:,2])
# m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84]), axis=0))

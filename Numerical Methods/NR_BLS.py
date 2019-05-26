from bls import blsprice
def bsput(sigma):
	S=112.6;
	K=100;
	r=0.02;
	T=2/12;
	return blsprice('P',S,K,r,sigma,T)
def newtstepBS(x1):
	P=5;#known from the market
	u=10**-5
	df=(bsput(x1+u)-bsput(x1))/u; #approximation for f'

	return x1-(bsput(x1)-P)/df;#x1-f(x1)/f'(x1)
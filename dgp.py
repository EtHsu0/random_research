import numpy as np

def generate_DGP(n, d, rho, DGP):
    if (DGP == "anticasual"):
        return generate_spurious_data_anticausal(n, d, rho)
    elif (DGP == "linear"):
        return generate_spurious_data_linear(n, d, rho)
    elif (DGP == "plusminus"):
        return generate_spurious_data_plusminus(n, d, rho)
    elif (DGP == "fortest"):
        return generate_spurious_data_fortest(n, d, rho)
    elif (DGP == "normal"):
        return generate_spurious_data(n, d, rho)


beta_gen = lambda d : np.array([1]*(d//2 + 1) + [0]*(d//2 - 1))
sigmoid = lambda x :  1/(1 + np.exp(-x))

def generate_spurious_data_anticausal(n, d, rho=0): 
    # anticausal is a type of data generating process where p(y,z) changes 
    # across train and test while p( X | Y, Z) stays the same
    rng = np.random
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d)
    noise = rng.normal(0.0, 0.5, size=n)
    beta = beta_gen(d)/np.sqrt(d)
  
    p_y_1_g_x = sigmoid(xs @ beta + rho) # p( y = 1 | X ) = sigma(beta^T x)

    ys = np.random.random(size=(n,)) < p_y_1_g_x # why does this sample from the bernoulli distirbution p( y = 1 | X )?

    p_z_1_g_x = sigmoid(rho*(2*ys - 1)) # p( z = 1 | X ) = sigma(beta_z^T x)
    
    zs = np.random.random(size=(n,)) < p_z_1_g_x # why does this work?

    xs[:, -1] = zs # This make model perfect as X and Y is perfect related

    # Z \not\perp Y \g X # this is wrong X--> Y --> Z, 
    
    # p(Y=y, Z=z| X) = 0 if  z\not= X[:, -1], do we agree? p(Y = y, Z=z, X=x) = 0 if z \not= x[:,-1], p(Y = y, Z=z, X=x) / p(X = x) = p(Y=y, Z=z| X)  = 0
    # p(Y=y| X) p(Z =z| X), p(Z =z| X)=0 when z\not= X[:, -1]
    # p(Y=y, Z=z| X) =  p(Y=y| X) p(Z =z| X) \implies Y \indep Z \g X

    # Y \not\indep Z \g X[:,0:d-1]

    return xs, ys, zs


def generate_spurious_data(n, d, rho):
    rng = np.random
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d)
    noise = rng.normal(0.0, 0.5, size=n)
    ys = [[rng.binomial(n=1, p=(np.prod(xs[i][:-5]) ** 0.2))] for i in range(n)]
    zs = [[rng.binomial(n=1, p=(np.prod(xs[i]) ** 0.1))] for i in range(n)]
    # print(xs.shape, len(ys))
    return xs, ys, zs


def generate_spurious_data_linear(n, d, rho=0.9):
    rng = np.random
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d)
    noise = rng.normal(0.0, 0.5, size=n)
    beta = beta_gen(d)
    beta_z = beta_gen(d)
    np.random.shuffle(beta_z) # randomize positions of z

    # zs = np.array([[rng.binomial(n=1, p=(np.prod(xs[i]) ** 0.1))] for i in range(n)]
  
    p_y_1_g_x = sigmoid(xs @ beta * rho) # p( y = 1 | X ) = sigma(beta^T x) = 1 / (1 + exp(beta^T X*rho))
    
    p_z_1_g_x = sigmoid(xs @ beta_z / rho) # p( z = 1 | X ) = sigma(beta_z^T x)

    # ys = [[rng.binomial(n=1, p=(np.prod(xs[i][:-5]) ** 0.2))] for i in range(n)]

    ys = np.random.random(size=(n,)) < p_y_1_g_x # why does this sample from the bernoulli distirbution p( y = 1 | X )?
    
    zs = np.random.random(size=(n,)) < p_z_1_g_x # why does this work?

    xs[:, -1] = zs

    # Y \perp Z | X, conditionally on X, Y and Z are independent
    
    # Y \not\perp Z, unconditionally or marginally
    return xs, ys, zs

def generate_spurious_data_fortest(n, d, rho=0.1):
    rng = np.random
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d)
    noise = rng.normal(0.0, 0.5, size=n)
    beta = beta_gen(d)
    beta_z = beta_gen(d)
    np.random.shuffle(beta_z) # randomize positions of z

    # zs = np.array([[rng.binomial(n=1, p=(np.prod(xs[i]) ** 0.1))] for i in range(n)]
  
    p_y_1_g_x = sigmoid(xs @ beta * rho) # p( y = 1 | X ) = sigma(beta^T x)
    
    p_z_1_g_x = sigmoid(xs @ beta_z / rho) # p( z = 1 | X ) = sigma(beta_z^T x)

    # ys = [[rng.binomial(n=1, p=(np.prod(xs[i][:-5]) ** 0.2))] for i in range(n)]

    ys = np.random.random(size=(n,)) < p_y_1_g_x # why does this sample from the bernoulli distirbution p( y = 1 | X )?
    
    zs = np.random.random(size=(n,)) < p_z_1_g_x # why does this work?

    xs[:, -1] = ys # This make model perfect as X and Y is perfect related

    # sigmoid(xs @ beta * rho) does not equal Y with probability 1
    return xs, ys, zs


def generate_spurious_data_plusminus(n, d, rho=0):
    rng = np.random
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d)
    noise = rng.normal(0.0, 0.5, size=n)
    beta = beta_gen(d)
    beta_z = beta_gen(d)
    np.random.shuffle(beta_z) # randomize positions of z

    # zs = np.array([[rng.binomial(n=1, p=(np.prod(xs[i]) ** 0.1))] for i in range(n)]
    logit_y_1_g_x = 5*(xs @ beta  - (xs @ beta).mean())
    p_y_1_g_x = sigmoid(logit_y_1_g_x + rho) # p( y = 1 | X ) = sigma(beta^T x)    
    
    p_z_1_g_x = sigmoid(xs @ beta_z - rho) # p( z = 1 | X ) = sigma(beta_z^T x)

    # ys = [[rng.binomial(n=1, p=(np.prod(xs[i][:-5]) ** 0.2))] for i in range(n)]

    ys = np.random.random(size=(n,)) < p_y_1_g_x # why does this sample from the bernoulli distirbution p( y = 1 | X )?
    
    zs = np.random.random(size=(n,)) < p_z_1_g_x # why does this work?

    xs[:, -1] = zs 

    return xs, ys, zs

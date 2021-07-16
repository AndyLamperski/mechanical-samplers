import torch as pt
import numpy as np
import numpy.random as rnd
import scipy.integrate as itg
import functools as ft

def projectHyperplane(x,a,b):
    """
    given x, find the closest y such that
    a'y <= b
    """
    r = a@x-b
    if r <= 0:
        return x
    else:
        lam = r/pt.sum(a**2)
        y = x-lam * a
        return y
    
def normalizeConstraints(A,b):
    nms = pt.norm(A,dim=1)
    A_normed = pt.diag(1./nms) @A
    b_normed = b /nms
    return A_normed,b_normed

def projectPolyhedron(x,A,b,tol=1e-6,normed=False):
    if not normed:
        A_normed,b_normed = normalizeConstraints(A,b)
    else:
        A_normed = A
        b_normed = b
    
    y = x.detach().clone()
    z = y.detach()
    
    
    gaps = A_normed @ y - b_normed 
    
    while pt.max(gaps) > tol:
    
        i = pt.argmax(gaps)
        z = projectHyperplane(y,A_normed[i],b_normed[i])
  
        y = z.detach().clone()
        gaps = A_normed@y - b_normed    
        
    return z

    

def reflectHyperplane(p,a):
    return p - (2. * p@a / (pt.sum(a**2))) * a

def reflectPolyhedron(x,A,b,tol=1e-6):
    n = int(len(x)/2)
    q = x[:n]
    p = x[n:]
    
    A_normed,b_normed = normalizeConstraints(A,b)
    q_proj = projectPolyhedron(q,A_normed,b_normed)
    
    gaps = A_normed @ q_proj - b_normed
    

    active = pt.where(gaps >= -tol)[0]
    A_a = A_normed[active]
    
    p_ref = p.detach().clone()
    
    
    violating = pt.where(A_a@p_ref > tol)[0]
    
    
    while len(violating) > 0:
        i = pt.randint(0,len(violating),size=(1,))[0]
        a = A_a[violating[i]]
        p_ref = reflectHyperplane(p_ref,a)
        violating = pt.where(A_a@p_ref > tol)[0]
        
    return pt.cat([q_proj,p_ref])

def splitState(x):
    n = int(len(x)/2)
    q = x[:n]
    p = x[n:]
    return q,p


class SDESampler:
    def __init__(self,f_fun,g_fun,eta,constraint_fun=None):
        self.f = f_fun
        self.g = g_fun
        self.eta = eta
        self.constrain = constraint_fun

    def step(self,x,*args):
        f = self.f(x,*args)
        g = self.g(x,*args)
        m = g.shape[1]
        w = pt.randn(m)
        x_next = x + self.eta * f + np.sqrt(self.eta) * g@w
        x_next = self.constrain(x_next,*args)
        return x_next.detach().clone().requires_grad_(True)


def torch_to_numpy(f,dt=pt.float32):
    def f_wrapped(x_np):
        x_pt = pt.tensor(x_np,dtype=dt,requires_grad=True)
        y_pt = f(x_pt)
        return y_pt.detach().numpy()

    return f_wrapped

class HybridSampler:
    def __init__(self,f_fun,init_fun,event_fun=None,reset_fun=None):
        vf = torch_to_numpy(f_fun)
        self.f = lambda t,y : vf(y)
        self.init = init_fun
        
        if event_fun is not None:
            ef = torch_to_numpy(event_fun)
            self.event = lambda t,y : ef(y)
            self.event.terminal = True
            self.event.direction = 1
        else:
            self.event = None
            
        if reset_fun is not None:
            self.reset = torch_to_numpy(reset_fun)
        else:
            self.reset = None

        
    def step(self,x,tstart=0.,T=1.):
        x = self.init(x)
        x0 = x.detach().numpy()

        finished = False
    
        Time_np_list = []
        X_np_list = []

        t_span = [tstart,tstart+T]
        while not finished:
            sol = itg.solve_ivp(self.f,t_span,x0,events = self.event)

            Time_np_list.append(sol.t)
            X_np_list.append(sol.y)
            
            if sol.status == 0:
                finished = True
            elif sol.status == 1:
                x0 = self.reset(sol.y[:,-1])
                t_span = [sol.t[-1],tstart + T]
            
            
        X_np = np.hstack(X_np_list).T
        Time_np = np.hstack(Time_np_list)
    
        X = pt.tensor(X_np,dtype=x.dtype)
        Time = pt.tensor(Time_np,dtype=x.dtype)
        return X,Time
        

def basicHamiltonsEquations(x,loss_fun,*args):
    q,p = splitState(x)
    x.grad = pt.zeros_like(x)
    
    n = len(q)
    loss = loss_fun(q,*args)
    loss.backward()
    return pt.cat([p,-x.grad[:n]])

    
class FOLangevin(SDESampler):
    def __init__(self,loss_fun,eta,A=None,b=None):

        def f_fun(x,*args):
            x.grad = pt.zeros_like(x)
            f = loss_fun(x,*args)
            f.backward()
            return -x.grad

        g_fun = lambda x,*args : np.sqrt(2) * pt.eye(len(x))

        
        def constrain(x,*args):
            if A is None:
                return x
            else:
                return projectPolyhedron(x,A,b)
            
        
        super().__init__(f_fun,g_fun,eta,constrain)
    # def step(self,x):
    #     x.grad = pt.zeros_like(x)
    #     f = self.loss(x)
    #     f.backward()
    #     x_next = x - self.eta * x.grad + np.sqrt(2*self.eta) * pt.randn_like(x)
    #     if self.A is not None:
    #         x_next = projectPolyhedron(x_next,self.A,self.b)
    #     return x_next.detach().clone().requires_grad_(True)


class SOLangevin(SDESampler):
    def __init__(self,loss_fun,eta,gamma=1.,A=None,b=None):
        def f_fun(x,*args):
            q,p = splitState(x)
            n = len(q)
            v = basicHamiltonsEquations(x,loss_fun,*args)

            return pt.cat([v[:n],v[n:]-gamma*p])

        def g_fun(x,*args):
            n = int(len(x)/2)
            return pt.cat([pt.zeros((n,n)),
                           np.sqrt(2)*pt.eye(n)],dim=0)

        def constrain(x,*args):
            if A is None:
                return x
            else:
                return reflectPolyhedron(x,A,b)


        super().__init__(f_fun,g_fun,eta,constrain)
            

class HMCSampler(HybridSampler):
    def __init__(self,loss_fun,A=None,b=None,tol=1e-3):
        def f_fun(x):
            return basicHamiltonsEquations(x,loss_fun)

        if A is None:
            event_fun = None
            reset_fun = None

        else:
            A_norm,b_norm = normalizeConstraints(A,b)
            def event_fun(x):
                n = int(len(x)/2)
                violation = pt.max(A_norm@x[:n]-b_norm)-tol
                return violation 

            
            def reset_fun(x):
                return reflectPolyhedron(x,A,b)

        def init_fun(x):
            q,p = splitState(x)
            x_new = pt.cat([q,pt.randn_like(p)])
            if A is None:
                return x_new
            else:
                return reflectPolyhedron(x_new,A,b)

        super().__init__(f_fun,init_fun,event_fun,reset_fun)


def genHamiltonsEquations(x,H_fun):
    x.grad = pt.zeros_like(x)
    q,p = splitState(x)

    n = len(q)
    H = H_fun(x)
    H.backward()
    q_dot = x.grad[n:]
    p_dot = -x.grad[:n]

    return pt.cat([q_dot,p_dot])

class RHMCSampler(HybridSampler):
    def __init__(self,loss_fun,g_inv_fun):
        def H_fun(x):
            q,p = splitState(x)
            g_inv = g_inv_fun(q)

            _,ld = pt.linalg.slogdet(g_inv)
            U= loss_fun(q) - .5*ld

            K = .5 * p@g_inv@p
            return U+K

        def f_fun(x):
            return genHamiltonsEquations(x,H_fun)

        def init_fun(x):
            q,p = splitState(x)
            g_inv = g_inv_fun(q)

            
            C = pt.linalg.cholesky(g_inv)
            p_new = pt.linalg.solve(C.T,pt.randn_like(p))
            return pt.cat([q,p_new])

        super().__init__(f_fun,init_fun)

    
    
class GibbsSampler:
    def __init__(self,samplers,dependents):
        """
        sampler = GibbsSampler(samplers,dependents)

        sampler performs gibbs sampling for a list of variables.
        
        sampler[i] draws samplers of x[i] 
        dependents[i] denotes the other variabls, x[j] that sampler i depends at
        """
        self.samplers = samplers
        self.dependents = dependents

    def step(self,x):
        """
        X should be 
        """

        x_next = []
        for xi,si,di in zip(x,self.samplers,self.dependents):
            x_dep = [x[j] for j in di]
            xi_next = si.step(xi,*x_dep)
            x_next.append(xi_next)

        return x_next
            
class FOLangevinSmoother(FOLangevin):
    def __init__(self,initLL,measLL,stepLL,eta,Y,A=None,b=None):
        NumSteps = len(Y)
        if A is None:
            A_traj = None
            b_traj = None
        else:
            A_traj = pt.kron(pt.eye(NumSteps),A)
            b_traj = pt.kron(pt.ones(NumSteps),b)

        def loss_fun(X):
            nX = int(len(X)/NumSteps)
            loss = -initLL(X[:nX])
            
            for i in range(NumSteps):
                x_cur = X[i*nX:(i+1)*nX]
                y = Y[i]
                loss = loss - measLL(x_cur,y)

                if i > 0:
                    x_prev = X[(i-1)*nX:i*nX]
                    loss = loss - stepLL(x_prev,x_cur)
                    
                if i < (NumSteps-1):
                    x_next = X[(i+1)*nX:(i+2)*nX]
                    loss = loss - stepLL(x_cur,x_next)

            return loss

        super().__init__(loss_fun,eta,A=A_traj,b=b_traj)

class FOLangevinSmoothID(FOLangevin):
    def __init__(self,initLL,measLL,stepLL,paramLL,
                 eta,Y,U=None,NumParam=None,NumState=None,
                 A_s=None,b_s=None,A_p=None,b_p=None):
        """
        NumParam - Optional if parameters are constrained. It must be given if 
                   if the parameters are unconstrained.  It is required to find the parameters                   from the full vector of states and parameters
        NumState - Optional integer. Only needed if the states are unconstrained but the
                   parameters are constrained. In this case, it is required to determine 
                   the shape of the constraint matrices. 

        """
        
        NumSteps = len(Y)
        
        if A_s is None:
            A_traj = None
            b_traj = None
        else:
            NumState = A_s.shape[1]
            A_traj = pt.kron(pt.eye(NumSteps),A_s)
            b_traj = pt.kron(pt.ones(NumSteps),b_s)

        if A_p is not None:
            NumParam = A_p.shape[1]
            
        if (A_traj is None):
            if A_p is None:
                A = None
                b = None
            else:
                # If 
                m = len(b_p)
                A = pt.cat([pt.zeros(m,NumState*NumSteps),A_p])
                b = b_p
        elif A_p is None:
            m = len(b_traj)
            A = pt.cat([A_traj,pt.zeros((m,NumParma))])
            b = b_traj
        else:
            A = pt.block_diag(A_traj,A_p)
            b = pt.cat([b_traj,b_p])

        def loss_fun(X):
            nX = NumState
            params = X[nX*NumSteps:]
            loss = -initLL(X[:nX],params) - paramLL(params)

            
            
            for i in range(NumSteps):
                x_cur = X[i*nX:(i+1)*nX]
                y = Y[i]
                if U is not None:
                    u_cur = U[i]
                    args_cur = (u_cur,params)
                else:
                    u_cur = None
                    args_cur = (params,)
                loss = loss - measLL(x_cur,y,*args_cur)

                if i > 0:
                    if U is not None:
                        u_prev = U[i-1]
                        args_prev = (u_prev,params)
                    else:
                        args_prev = (params,)
                        
                    x_prev = X[(i-1)*nX:i*nX]
                    loss = loss - stepLL(x_prev,x_cur,*args_prev)
                    
                if i < (NumSteps-1):
                    x_next = X[(i+1)*nX:(i+2)*nX]
                    loss = loss - stepLL(x_cur,x_next,*args_cur)

            return loss

        super().__init__(loss_fun,eta,A=A,b=b)

class SparseSGLangevin:
    def __init__(self,loss_fun_list,var_dict,eta,dim_list = None,A_list=None,b_list=None):

        if dim_list is None:
            dim_list = [A.shape[1] for A in A_list]

        M = len(loss_fun_list)


        fun_dict = {}

        for i in range(M):
            neighbors = var_dict[i]

            for j in neighbors:
                if j not in fun_dict.keys():
                    fun_dict[j] = [i]
                else:
                    fun_dict[j].append(i)

        
        fun_dict = {i : sorted(fun_dict[i]) for i in sorted(fun_dict)}


        
        
        if A_list is None:
            A_list = [None for _ in range(M)]
            b_list = [None for _ in range(M)]

        G_mat_list = []
        f_fun_list = []
        g_fun_list = []
        constrain_list = []
        self.samplers = []
        for i in range(M):
            neighbors = var_dict[i]


            
            chol_list = []

            for j in neighbors:
                dim_j = dim_list[j]
                oc_j = len(fun_dict[j])

                chol = np.sqrt(2*M/oc_j) * pt.eye(dim_j)
                chol_list.append(chol)

            G_mat = pt.block_diag(*chol_list)

            G_mat_list.append(G_mat)
        
        def f_fun(z,i):
            z.grad = pt.zeros_like(z)
            f = loss_fun_list[i](z)
            f.backward()
            return -M*z.grad

            
        def g_fun(z,i):
            return G_mat_list[i]

            
        def constrain(x,i):
            if A_list[i] is None:
                return x
            else:
                return projectPolyhedron(x,A_list[i],b_list[i])

        self.sampler = SDESampler(f_fun,g_fun,eta,constrain)

        self.var_dict = var_dict
        self.dim_list = dim_list

    def step(self,z_list):
        M =len(self.var_dict)
        J = rnd.randint(M)
        neighbors = self.var_dict[J]

        z_cur_list = []

        cur_dims = []

        
        
        for j in neighbors:
            zj = z_list[j]
            z_cur_list.append(zj)
            cur_dims.append(len(zj))

        z = pt.cat(z_cur_list).detach().clone().requires_grad_(True)


        z_next = self.sampler.step(z,J)

        strides = np.hstack([[0],np.cumsum(cur_dims)])
        for j_num,j in enumerate(neighbors):
            zj_next = z_next[strides[j_num]:strides[j_num+1]]
            z_list[j] = zj_next.detach().clone().requires_grad_(True)

        return z_list
        
        

def logLikeToLoss(fun):
    loss = lambda x,*args : -fun(x,*args)
    return loss

    
class SGSmoothID(SparseSGLangevin):
    def __init__(self,initLL,measLL,stepLL,paramLL,
                 eta,Y,U=None,NumParam=None,NumState=None,
                 A_s=None,b_s=None,A_p=None,b_p=None):

                
        NumSteps = len(Y)

        N = NumSteps+1
        var_lists = [[N-1],[0,N-1]] + [[i,i+1,N-1] for i in range(NumSteps-1)] + \
                    [[i,N-1] for i in range(NumSteps)]

        var_dict = { i : neighbors for i,neighbors in enumerate(var_lists)}
        


        if A_s is None:
            if A_p is None:
                A_meas = None
                b_meas = None

                A_step = None
                b_step = None
            else:
                A_meas = A_p.detach().clone()
                b_meas = b_p.detach().clone()

                A_step = A_p.detach().clone()
                b_step = b_p.detach().clone()

                NumParam = A_p.shape[1]
        elif A_p is None:
            A_meas = A_s.detach().clone()
            b_meas = b_s.detach().clone()

            A_step = pt.block_diag(A_s,A_s)
            b_step = pt.cat([b_s,b_s])

            NumState = A_s.shape[1]
        else:
            A_meas = pt.block_diag(A_s,A_p)
            b_meas= pt.cat([b_s,b_p])

            A_step = pt.block_diag(A_s,A_s,A_p)
            b_step = pt.cat([b_s,b_s,b_p])

            NumParam = A_p.shape[1]
            NumState = A_s.shape[1]
                           
            
        A_list = [A_p,A_meas] + [A_step for _ in range(NumSteps-1)] + \
                 [A_meas for _ in range(NumSteps)]

        b_list = [b_p,b_meas] + [b_step for _ in range(NumSteps-1)] + \
                 [b_meas for _ in range(NumSteps)]

                            
        dim_list = NumSteps * [NumState] + [NumParam]


        paramLoss = logLikeToLoss(paramLL)
        def initLoss(z):
            x = z[:NumState]
            theta = z[NumState:]
            return -initLL(x,theta)

        stepLosses = []
        measLosses = []

        if U is None:
            U = NumSteps * [None]

        for i in range(NumSteps):
            y = Y[i]

            
            if U is not None:
                u = U[i]
            else:
                u = None

        def measLoss(z,i):
            x = z[:NumState]
            theta = z[NumState:]
            y = Y[i]
            u = U[i]
            if u is not None:
                args = (u,theta)
            else:
                args = (theta,)

            return -measLL(x,y,*args)


        measLosses = [ft.partial(measLoss,i=j) for j in range(NumSteps)]

        def stepLoss(z,i):
            x_cur = z[:NumState]
            x_next = z[NumState:2*NumState]
            theta = z[2*NumState:]

            u = U[i]
            if u is not None:
                args = (u,theta)
            else:
                args = (theta,)
            return -stepLL(x_cur,x_next,*args)


        stepLosses = []
        for j in range(NumSteps-1):
            stepLosses.append(ft.partial(stepLoss,i=j))

        loss_fun_list = [paramLoss,initLoss] + stepLosses + measLosses

        
        super().__init__(loss_fun_list,var_dict,eta,dim_list,A_list,b_list)
        
class SGLangevin:
    def __init__(self,loss_fun_list,eta,A=None,b=None):
        M = len(loss_fun_list)

        
        def f_fun(z,i):
            z.grad = pt.zeros_like(z)
            f = loss_fun_list[i](z)
            f.backward()
            return -M*z.grad

        def g_fun(z,*args):
            return np.sqrt(2*M) * pt.eye(len(z))


        def constrain(x,*args):
            if A is None:
                return x
            else:
                return projectPolyhedron(x,A,b)

        self.sampler = SDESampler(f_fun,g_fun,eta,constrain)

        self.M = M
    def step(self,x):
        J = rnd.randint(self.M)
        x_next = self.sampler.step(x,J)
        return x_next.detach().clone().requires_grad_(True)
        
class ARLangevin(SGLangevin):
    def __init__(self,initLL,arLL,order,eta,Y,pad=None,A=None,b=None):
        if pad is None:
            pad = order

        

        initLoss = lambda x : -initLL(x)

        def predictionLoss(theta,i):
            y = Y[i]
            H = Y[i-order:i]
            return -arLL(y,H,theta)

        
        predictionLosses = [ft.partial(predictionLoss,i=j) for j in range(pad,len(Y))]

        loss_list = [initLoss] + predictionLosses
        super().__init__(loss_list,eta,A,b)

        self.loss_list = loss_list

    def loglikelihood(self,theta):
        loss = pt.sum(pt.stack([loss(theta) for loss in self.loss_list]))
        return -loss
        

        
        

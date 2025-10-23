import numpy as np

def Secant_method(f, a, b, tol):
  """searches for a root of the function f in the interval [a, b]"""
  x0=a    #starting value 1
  x1=b    #starting value 2

  f0 = f(x0)
  f1 = f(x1)
  dx=-f1/(f1-f0)*(x1-x0)
  x1 = x1+dx

  while(abs(f1)>tol):
      f0=f1
      f1 = f(x1)
      dx = -f1/(f1-f0)*dx
      x1 = x1+dx
  
  return x1

def Bisection(f, a, b, tol):
  """searches for a root of the function f in the interval [a, b]"""
  x0=a
  x1=b
  
  f0=f(x0)
  f1=f(x1)

  if np.sign(f0)==np.sign(f1) :
    print("invalid starting values")
    return np.nan

  xm=0.5*(x0+x1)
  
  while(abs(f1)>tol):
    fm=f(xm)
    if np.sign(fm)==np.sign(f0) :
      f0=fm
    else:
      f1=fm
    xm=0.5*(x0+x1)
  
  return xm


def Euler(
    f: Callable[[float, NDArray], NDArray],
    x0: NDArray,
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray, NDArray, dict[str, int]]:
    """Eulers method for numerical solving of ODEs"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print("final step not hitting t_max exactly")

    info: dict[str, int] = dict(n_feval=0)
    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    t[0] = t0
    x[:, 0] = x0
    for i in range(steps):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        x[:, i + 1 : i + 2] = x[:, i : i + 1] + h * f(t[i], x[:, i : i + 1])
        info["n_feval"] += 1
    return t, x, info

def Midpoint(f, x0, t_max, h, t0=0):
  """Explicit Midpoint method for numerical solving of ODEs of the form x_dot=f(t,x)"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.expand_dims(np.linspace(t0,t_max,steps),axis=0)
  x=np.zeros((x0.shape[0], steps))
  t[:,0]=t0
  x[:,:1]=x0
  for i in range(steps-1):
    # t[:,i+1:i+2]=t[:, i:i+1]+h
    x[:,i+1:i+2]=x[:,i:i+1]+h*f(t[:, i:i+1]+h/2, x[:,i:i+1]+h/2*f(t[:, i:i+1], x[:,i:i+1]))

  return t,x


def AB2(f, x0, t_max, h, t0=0):
  """Adams Bashforth of order 2"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.expand_dims(np.linspace(t0,t_max,steps),axis=0)
  x=np.zeros((x0.shape[0], steps))
  _, x[:,:2]=Midpoint(f, x0, t0+h, h, t0)

  for i in range(1,steps-1):
    # t[:,i+1:i+2]=t[:, i:i+1]+h
    x[:,i+1:i+2]=x[:,i:i+1]+h/2*(3*f(t[:, i:i+1],x[:, i:i+1])-f(t[:, i-1:i],x[:,i-1:i]))

  return t,x

def AB3(f, x0, t_max, h, t0=0):
  """Adams Bashforth of order 3, first values calculated with Midpoint and AB2"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.expand_dims(np.linspace(t0,t_max,steps),axis=0)
  x=np.zeros((x0.shape[0], steps))
  _, x[:,:3]=AB2(f, x0, t0+2*h, h, t0)

  for i in range(2,steps-1):
    # t[:,i+1:i+2]=t[:, i:i+1]+h
    x[:,i+1:i+2]=x[:,i:i+1]+h/12*(23*f(t[:, i:i+1],x[:, i:i+1])-16*f(t[:,i-1:i],x[:, i-1:i])+5*f(t[:,i-2:i-1],x[:,i-2:i-1]))

  return t,x

def PECE(f, x0, t_max, h, nrep=1, t0=0):
  """PECE Method using AB3, AM4, starting with Midpoint and AB2"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.expand_dims(np.linspace(t0,t_max,steps),axis=0)
  x=np.zeros((x0.shape[0], steps))
  # t[:,:3], x[:,:3]=AB2(f, x0, t0+2*h, h, t0)
  _, x[:,:3]=AB2(f, x0, t0+2*h, h, t0)

  for i in range(2,steps-1):
    #t[:,i+1:i+2]=t[:, i:i+1]+h
    x[:,i+1:i+2]=x[:,i:i+1]+h/12*(23*f(t[:,i:i+1],x[:,i:i+1])-16*f(t[:,i-1:i],x[:,i-1:i])+5*f(t[:,i-2:i-1],x[:,i-2:i-1])) #AB3 predict/evaluate
    k=x[:,i:i+1]+h/24*(19*f(t[:,i:i+1],x[:,i:i+1])-5*f(t[:,i-1:i],x[:,i-1:i])+f(t[:,i-2:i-1],x[:,i-2:i-1]))
    for j in range(nrep):
      x[:,i+1:i+2]=k + 9*h/24*f(t[:,i+1:i+2],x[:,i+1:i+2]) #AM4 correct/evaluate loop

  return t,x


#######################   UNTESTED   ######################
def PECE(f, x0, t_max, h, t0=0, tol=1e-4):
  """PECE Method using AB3, AM4, starting with Midpoint and AB2, iterates until convergence with tolerance tol is met"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.expand_dims(np.linspace(t0,t_max,steps),axis=0)
  x=np.zeros((x0.shape[0], steps))
  _, x[:,:3]=AB2(f, x0, t0+2*h, h, t0)

  for i in range(2,steps-1):
    #t[:,i+1:i+2]=t[:, i:i+1]+h
    x[:,i+1:i+2]=x[:,i:i+1]+h/12*(23*f(t[:,i:i+1],x[:,i:i+1])-16*f(t[:,i-1:i],x[:,i-1:i])+5*f(t[:,i-2:i-1],x[:,i-2:i-1])) #AB3 predict/evaluate
    k=x[:,i:i+1]+h/24*(19*f(t[:,i:i+1],x[:,i:i+1])-5*f(t[:,i-1:i],x[:,i-1:i])+f(t[:,i-2:i-1],x[:,i-2:i-1]))
    
    last=np.Infinity
    while(np.linalg.norm(x[:,i+1:i+2]-last, ord=np.inf)>tol):
      last=x[:,i+1:i+2]
      x[:,i+1:i+2]=k + 9*h/24*f(t[:,i+1:i+2],x[:,i+1:i+2]) #AM4 correct/evaluate loop

  return t,x

def RK4(
    f: Callable[[float, NDArray], NDArray],
    x0: NDArray,
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray, NDArray, dict[str, int]]:
    """Classical Runge-Kutta Method, order 4"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print("final step not hitting t_max exactly")

    info: dict[str, int] = dict(n_feval=0)

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    t[0] = t0
    x[:, 0] = x0
    for i in range(steps):
        k1 = f(t[i], x[:, i : i + 1])
        k2 = f(t[i] + 0.5 * h, x[:, i : i + 1] + 0.5 * h * k1)
        k3 = f(t[i] + 0.5 * h, x[:, i : i + 1] + 0.5 * h * k2)
        k4 = f(t[i] + h, x[:, i : i + 1] + h * k3)
        x[:, i + 1 : i + 2] = x[:, i : i + 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        info["n_feval"] += 4
    return t, x, info

def DP45(f, t_max, x0, t0=0, eps=10**-5, s_clip=[0.2, 5], tol_safety=0.9):
  """Dormand Prince 5(4) Method, MATLAB ode45 solver"""
  h=tol_safety*eps**(1/5)/4

  t=[t0]
  x=[x0]
  t_crit=[]
  x_crit=[]

  k=0
  k1=f(t[0],x[0 ])
  while t[k] < t_max: #iterate until t_max is reached
    if(t[k]+h>t_max): #shorten h if we would go further than necessary
      h=t_max-t[k]

    t_pred=t[k]+h
    #calulate k's
    #k1=f(t[k], x[k])
    k2=f(t[k]+1/5*h, x[k]+1/5*h*k1)
    k3=f(t[k]+3/10*h, x[k]+h*(3*k1+9*k2)/40)
    k4=f(t[k]+4/5*h, x[k]+h*(44/45*k1-56/15*k2+32/9*k3))
    k5=f(t[k]+8/9*h, x[k]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4))
    k6=f(t_pred, x[k]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5))

    x_pred=x[k]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6) #embedding, as f(x_pred)==k2
    k2=f(t_pred,x_pred) #reusing k2 instead of new variable k7, as it is not used again

    err=h*np.linalg.norm(71/57600*k1-71/16695*k3+71/1920*k4-17253/339200*k5+22/525*k6-1/40*k2, ord=np.inf)#h*np.linalg.norm(71/57600*k1-71/16695*k3+71/1920*k4-17253/339200*k5+22/525*k6-1/40*k2, ord=np.inf) #local error:||x_pred-X_pred7||
    if(err==0): #set s to max tp prevent divide by zero error
      s=s_clip[1]
    else:
      s=np.clip(tol_safety*(eps/err)**(1/5), s_clip[0], s_clip[1]) # s gets clipped to prevent to extreme changes of h, also err might become zero

    if(err<eps):#or h==h_min): #accept result if tolerance is met, or we cant decrease h anymore
        k1=k2
        k+=1
        x.append(x_pred)
        t.append(t_pred)
    else:
        if(s>1):
            print("h gets increased to decrease error? This shouldnt happen!")
        t_crit.append(t_pred)
        x_crit.append(x_pred)

    h=h*s

    #h=np.clip(h,h_min,h_max)
  return t,x, (t_crit, x_crit)


def BS23(f, t_max, x0, t0=0, eps=10**-5, s_clip=[0.2, 5], tol_safety=0.9):
  """Bogacki–Shampine Method, MATLAB ode23 solver"""
  h=tol_safety*eps**(1/3)/4 #t_max-t0

  t=[t0]
  x=[x0]
  t_crit=[]
  x_crit=[]

  k=0
  k1=f(t[0],x[0 ])
  while t[k] < t_max: #iterate until t_max is reached
    if(t[k]+h>t_max): #shorten h if we would go further than necessary
      h=t_max-t[k]

    t_pred=t[k]+h
    #calulate k's
    k2=f(t[k]+1/2*h, x[k]+1/2*h*k1)
    k3=f(t[k]+3/4*h, x[k]+3/4*h*k2)
    x_pred=x[k]+h*(2*k1+3*k2+4*k3)/9
    k4=f(t_pred,x_pred)

    err=h*np.linalg.norm(-5/72*k1+1/12*k2+1/9*k3-1/8*k4, ord=np.inf)#h*np.linalg.norm(71/57600*k1-71/16695*k3+71/1920*k4-17253/339200*k5+22/525*k6-1/40*k2, ord=np.inf) #local error:||x_pred-X_pred7||
    if(err==0): #set s to max tp prevent divide by zero error
      s=s_clip[1]
    else:
      s=np.clip(tol_safety*(eps/err)**(1/3), s_clip[0], s_clip[1]) # s gets clipped to prevent to extreme changes of h, also err might become zero

    if(err<eps):#or h==h_min): #accept result if tolerance is met, or we cant decrease h anymore
        k1=k4
        k+=1
        x.append(x_pred)
        t.append(t_pred)
    else:
        if(s>1):
            print("h gets increased to decrease error? This shouldnt happen!")
        t_crit.append(t_pred)
        x_crit.append(x_pred)

    h=h*s

    #h=np.clip(h,h_min,h_max)
  return t,x, (t_crit, x_crit)

def Backwards_Euler(
    f: Callable[[float, NDArray], NDArray],
    x0: NDArray,
    t_max: float,
    h: float,
    t0: float = 0.0,
    solvertol: float = 1e-5,
) -> tuple[NDArray, NDArray, dict[str, int]]:
    """Backwards Euler Method. System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print("final step not hitting t_max exactly")

    info: dict[str, int] = dict(n_feval=0)

    t = np.zeros((steps + 1,), dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    t[0] = t0
    x[:, 0] = x0

    for i in range(steps):
        t[i + 1] = t[i] + h
        f_imp = lambda x_next: x_next - x[:, i : i + 1] - h * f(t[i + 1], x_next)
        # x[:, i + 1 : i + 2] = solver(
        #     f_imp, x[:, i : i + 1], np.eye(x0.shape[1]), tol=solvertol
        # )
        sol = root(f_imp, x0=x[:, i : i + 1], tol=solvertol, method="hybr")
        x[:, i + 1 : i + 2] = sol.x
        if not sol.success:
            print("solver did not converge")
            exit()
        info["n_feval"] += sol.nfev

    return t, x, info

def BDF2(f, x0, t_max, h, t0=0, solver=BFGS, solvertol=1e-5):
  """Backward differantiation Formula of order 2 for stiff systems.
  Starting values generated with backwards Euler method
  System of Equations solved by solver(f==0, a, b, tol)"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.zeros((1, steps))
  x=np.zeros((x0.shape[0], steps))
  t[:,:2], x[:,:2]=Backwards_Euler(f, x0, t0+h, h, t0, solver, solvertol)

  for i in range(steps-1):
    t[:,i+1:i+2]=t[:, i:i+1]+h
    f_imp=lambda x_next: x_next-4/3*x[:,i:i+1]+1/3*x[:,i-1:i]-2/3*h*f(t[:,i+1:i+2], x_next)
    x[:,i+1:i+2]=solver(f_imp, x[:,i:i+1], np.eye(x0.shape[1]), solvertol)
  return t,x

def BDF3(f, x0, t_max, h, t0=0, solver=BFGS, solvertol=1e-5):
  """Backward differantiation Formula of order 3 for stiff systems.
  Starting values generated with backwards Euler method and BDF2
  System of Equations solved by solver(f==0, a, b, tol)"""
  steps=np.math.ceil((t_max-t0)/h)+1
  t=np.zeros((1, steps))
  x=np.zeros((x0.shape[0], steps))
  t[:,:3], x[:,:3]=BDF2(f, x0, t0+2*h, h, t0, solver, solvertol)

  for i in range(steps-1):
    t[:,i+1:i+2]=t[:, i:i+1]+h
    f_imp=lambda x_next: x_next-18/11*x[:,i:i+1]+9/11*x[:,i-1:i]-2/11*x[:,i-1:i-1]-6/11*h*f(t[:,i+1:i+2], x_next)
    x[:,i+1:i+2]=solver(f_imp, x[:,i:i+1], np.eye(x0.shape[1]), solvertol)
  return t,x

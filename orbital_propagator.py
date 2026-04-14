import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
class vettore():
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
    def __mul__(self, scalar):
        return vettore(self.x*scalar, self.y*scalar, self.z*scalar)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __add__(self, v2):
        return vettore(self.x+v2.x, self.y+v2.y, self.z+v2.z)
    def dot(self,v2):
        v1=np.array([self.x,self.y,self.z])
        arr2=np.array([v2.x,v2.y,v2.z])
        result=np.dot(v1,arr2)
        return result     
    def norm(self):
        v1=np.array([self.x,self.y,self.z])
        result=np.sqrt(np.dot(v1,v1))
        return result
    def cross(self,v2):
        v1=np.array([self.x,self.y,self.z])
        arr2=np.array([v2.x,v2.y,v2.z])
        crossvec=np.cross(v1,arr2)
        result=vettore(crossvec[0],crossvec[1],crossvec[2])
        return result
    def angle(self,v2):
        result=np.degrees(np.arccos((self.dot(v2))/(self.norm()*v2.norm())))
        return result
    def __str__(self):
        return (f"Componente x:{self.x}\nComponente y:{self.y}\nComponente z:{self.z}\n")
def orbital_state(r,v,mu):
    """
    Identifica i parametri orbitali ricevendo in input il raggio, la velocità e il parametro gravitazionale
    Input: Raggio in km, velocità in km/s, parametro gravitazionale in km^3/s^2
    Output:Calcola i parametri orbitali h, eps ed e
    """
    h=r.cross(v)
    eps=(1/2)*((v.norm())**2)-mu/r.norm()
    e=vettore((1/mu)*v.cross(h).x-r.x/r.norm(),(1/mu)*v.cross(h).y-r.y/r.norm(),(1/mu)*v.cross(h).z-r.z/r.norm())
    stato=[h,eps,e]
    return stato
def classify_orbit(eps,e): 
    """
    Identifica la conica lungo la quale si svolge l'orbita ricevendo come parametri l'energia specifica e l'eccentricità dell'orbita trovati con la funzione calcola_stato
    Input: eps,e calcolati con calcola_stato
    Output: Stringa che indica la conica lungo la quale si svolge l'orbita
    """
    if eps<0: 
        if e.norm()==0:
            result="Circonferenza"
        elif 0<e.norm()<1:
            result="Ellisse"
    elif eps==0:
        result="Parabola"
    else: result="Iperbole"
    return result
def plot_vectors(r,v,h,eps,e):
    """
    Mostra la direzione dei vettori normalizzati r,v,h ed e trovati con la funzione calcola_stato
    Input: Raggio in km, velocità in km/s, h,eps,e calcolati con calcola_stato
    Output:Mostra la direzione dei vettori normalizzati r,v,h ed e
    """
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.quiver(0,0,0,h.x,h.y,h.z,label="h",color='y',normalize=True)
    ax.quiver(0,0,0,r.x,r.y,r.z,label="r",color='b',normalize=True)
    ax.quiver(0,0,0,v.x,v.y,v.z,label="v",color='r',normalize=True)
    ax.quiver(0,0,0,e.x,e.y,e.z,label="e",color='k',normalize=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{classify_orbit(eps,e)}")
    ax.legend()
    plt.show()
def stumpff_S(z):
    """
    Calculates Stumpff's function S(z)
    """
    if z>0:
        S=(np.sqrt(z)-np.sin(np.sqrt(z)))/np.sqrt(z**3)
    elif z==0:
        S=1/6
    else: 
        S=(np.sinh(np.sqrt(-z))-np.sqrt(-z))/np.sqrt(z**3)
    return S
def stumpff_C(z):
    """
    Calculates Stumpff's function C(z)
    """
    if z>0:
        C=(1-np.cos(np.sqrt(z)))/z
    elif z==0:
        C=1/2
    else: 
        C=(np.cosh(np.sqrt(-z))-1)/(-z)
    return C
def cartesian_to_keplerian(r_0,v_0,mu):
    """
    Calculates the six orbital parameters, receiving as inputs the initial radius and velocity, and the gravitational parameter
    """
    h=orbital_state(r_0,v_0,mu)[0]
    e=orbital_state(r_0,v_0,mu)[2]
    i=np.rad2deg(np.arccos(h.z/h.norm()))
    print(i)
    Z=vettore(0,0,1)
    N=Z.cross(h)
    if N.norm() < 1e-3:
        Omega = 0
    else:
        Omega=np.rad2deg(np.arccos(N.x/N.norm()))
        if N.y<0:
            Omega=360-Omega
            
    print(e.norm())
    if e.norm()< 1e-3:
        omega = 0
    else:
        omega=np.rad2deg(np.arccos(N.dot(e)/N.norm()/e.norm()))
        if e.z<0:
            omega=360-omega
    if e.norm() < 1e-5:
        theta = 0
    else:
        theta = np.rad2deg(np.arccos(r_0.dot(e)/r_0.norm()/e.norm()))
        v_r = r_0.dot(v_0)/r_0.norm()
        if v_r < 0:
            theta = 360-theta  
        return h,e,i,Omega,omega,theta
def R1(phi):
        phi=np.deg2rad(phi)
        R1=np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
        return R1
def R3(phi):
        phi=np.deg2rad(phi)
        R3=np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
        return R3
def rot_mat(i,Omega,omega):
        Q=np.dot(R3(Omega),np.dot(R1(i),R3(omega)))
        return Q
def keplerian_to_cartesian(h,e,i,Omega,omega,theta,mu):
        """
        Calculates the radius and the velocity, in respect to the geocentric referance frame receveing as inputs the six orbital parameter and the gravitational parameter
        """
        theta=np.deg2rad(theta)
        rp=h.norm()**2/mu*(1/(1+e.norm()*np.cos(theta)))*np.cos(theta)
        rq=h.norm()**2/mu*(1/(1+e.norm()*np.cos(theta)))*np.sin(theta)
        r_per=np.array([rp,rq,0])
        vp=-mu/h.norm()*np.sin(theta)
        vq=mu/h.norm()*(e.norm()+np.cos(theta))
        v_per=np.array([vp,vq,0])
        Q=rot_mat(i,Omega,omega)
        r_geo=np.dot(Q,r_per)
        v_geo=np.dot(Q,v_per)
        return r_geo,v_geo
def plot_orbit():
    """
    Plots the orbit, using the keplerian_to_cartesian function
    """
    theta=np.linspace(0,360,2000)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    xs,ys,zs=[],[],[]
    for theta in np.linspace(0,360,2000):
        xs.append(keplerian_to_cartesian(risultati[0],risultati[1],risultati[2],risultati[3],risultati[4],theta,mu)[0][0])
        ys.append(keplerian_to_cartesian(risultati[0],risultati[1],risultati[2],risultati[3],risultati[4],theta,mu)[0][1])
        zs.append(keplerian_to_cartesian(risultati[0],risultati[1],risultati[2],risultati[3],risultati[4],theta,mu)[0][2])
    ax.clear()
    ax.plot(xs,ys,zs,label="orbita",linestyle='--',color='k')
    ax.plot(0,0,0,'bo',label="terra",markersize='10')
    ax.set_title("Orbita")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()
def update(frame):
    global ax
    ax.cla()
    r_max = max(max(xs), max(ys), max(zs))
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)
    ax.plot(xs[:frame],ys[:frame],zs[:frame],color='k',linestyle='--')
    ax.plot(xs[frame],ys[frame],zs[frame],'ro',label="satellite")
    ax.plot(0,0,0,'bo',markersize=10,label="terra")
    ax.set_title("Orbita")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    return ax
def animate_orbit():
    """
    Similar to the plot_orbit function, but the animation is not steay but dynamic
    """
    ani=FuncAnimation(fig,update,frames=range(2000),interval=1,repeat=False)
    plt.show()
    return 
if __name__ == "__main__":
    r_0 = vettore(7000, 0, 0)
    v_0 = vettore(0,-7.546,0)
    mu = 398600
    risultati = cartesian_to_keplerian(r_0, v_0, mu)
    xs,ys,zs=[],[],[]
    for theta in np.linspace(0,360,2000):
        xs.append(keplerian_to_cartesian(risultati[0],risultati[1],risultati[2],risultati[3],risultati[4],theta,mu)[0][0])
        ys.append(keplerian_to_cartesian(risultati[0],risultati[1],risultati[2],risultati[3],risultati[4],theta,mu)[0][1])
        zs.append(keplerian_to_cartesian(risultati[0],risultati[1],risultati[2],risultati[3],risultati[4],theta,mu)[0][2]) 
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    plot_orbit()

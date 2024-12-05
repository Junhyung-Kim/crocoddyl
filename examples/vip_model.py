# Display the solution
import numpy as np
import crocoddyl


class DifferentialActionModelCartpole(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 4, 4)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)

        self.m = 100
        self.g = 9.81
        self.costWeights = [1., 1., 5., 5.]  # sin, 1-cos, x, xdot, thdot, f

    def calc(self, data, x, u=None):
        if u is None: u = model.unone
        # Getting the state and control variables
        #x, xdot, px, My, y, ydot, py, Mx, z, zdot, p_z
        #pdotx, Mdoty, pdoty, Mdotx, zddot, pdot_z

        x_c, xdot, px, My, y, ydot, py, Mx, z, zdot, p_z = np.asscalar(x[0]), np.asscalar(x[1]), np.asscalar(x[2]), np.asscalar(x[3]), np.asscalar(x[4]), np.asscalar(x[5]), np.asscalar(x[6]), np.asscalar(x[7]), np.asscalar(x[8]), np.asscalar(x[9]), np.asscalar(x[10])
        pdotx, Mdoty, pdoty, Mdotx, zddot, pdot_z = np.asscalar(u[0]),np.asscalar(u[1]),np.asscalar(u[2]), np.asscalar(u[3]),np.asscalar(u[4]),np.asscalar(u[5])

        # Shortname for system parameters
        m, g = self.m, self.g
        
        # Defining the equation of motions
        xddot = (g+zddot)/(z-p_z)*(x_c-px-Mdoty/(m*(g+zddot)))
        yddot = (g+zddot)/(z-p_z)*(y-py+Mdotx/(m*(g+zddot)))
        data.xout = np.matrix([xddot, yddot]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([px, py-0.1025, Mx, My])).T
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


# Creating the DAM for the cartpole
cartpoleDAM = DifferentialActionModelCartpole()
cartpoleData = cartpoleDAM.createData()
cartpoleDAM = model = DifferentialActionModelCartpole()

# Using NumDiff for computing the derivatives. We specify the
# withGaussApprox=True to have approximation of the Hessian based on the
# Jacobian of the cost residuals.
cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)

# Getting the IAM using the simpletic Euler rule
timeStep = 5e-3
cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)

# Creating the shooting problem
x0 = np.array([0., 0. , 0., 0., 0., 0., 0., 0., 0.72, 0., 0.])
T = 50

terminalCartpole = DifferentialActionModelCartpole()
terminalCartpoleDAM = crocoddyl.DifferentialActionModelNumDiff(terminalCartpole, True)
terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM)

terminalCartpole.costWeights[0] = 1
terminalCartpole.costWeights[1] = 1
terminalCartpole.costWeights[2] = 5.
terminalCartpole.costWeights[3] = 5
problem = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)

# Solving it using DDP
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
ddp.solve([], [], 300)

for i in range(0, T):
    print(i)
    print(ddp.xs[i])
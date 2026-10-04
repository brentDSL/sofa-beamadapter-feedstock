"""Headless runtime check: SofaPython3 + BeamAdapter actually simulate.

Import tests only prove the extension modules load. This builds the
SingleBeamDeployment scene from BeamAdapter's examples, drives the deployment
from a Python Controller (exercising the pybind11 trampoline for virtual
overrides), reads state back through numpy views of Data fields, and steps the
solver. It fails if the plugin does not load, the scene does not initialise,
the controller callback never fires, or the solution goes non-finite.
"""
import sys

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime  # noqa: F401  (registers the default plugin search paths)

STEPS = 40


class Deploy(Sofa.Core.Controller):
    def __init__(self, ctrl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl = ctrl
        self.calls = 0

    def onAnimateBeginEvent(self, event):
        self.calls += 1
        with self.ctrl.xtip.writeableArray() as xtip:
            xtip[0] += 2.0


def create_scene(root):
    root.addObject("RequiredPlugin", pluginName=[
        "BeamAdapter",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Topology.Container.Dynamic",
    ])
    root.addObject("DefaultAnimationLoop")
    root.dt = 0.01
    root.gravity = [0.0, -9.81, 0.0]

    topo = root.addChild("EdgeTopology")
    topo.addObject("RodStraightSection", name="StraightSection", length=980.0, radius=0.9,
                   nbBeams=30, nbEdgesCollis=30, nbEdgesVisu=196,
                   youngModulus=20000, massDensity=0.00000155)
    topo.addObject("RodSpireSection", name="SpireSection", length=20.0, radius=0.9,
                   nbBeams=5, nbEdgesCollis=5, nbEdgesVisu=4, spireDiameter=25, spireHeight=0,
                   youngModulus=20000, massDensity=0.00000155)
    topo.addObject("WireRestShape", name="BeamRestShape", template="Rigid3d",
                   wireMaterials="@StraightSection @SpireSection")
    topo.addObject("EdgeSetTopologyContainer", name="meshLinesBeam")
    topo.addObject("EdgeSetTopologyModifier", name="Modifier")
    topo.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
    topo.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")

    beam = root.addChild("BeamModel")
    beam.addObject("EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1)
    beam.addObject("BTDLinearSolver", verification=False, subpartSolve=False, verbose=False)
    beam.addObject("RegularGridTopology", name="MeshLines", nx=61, ny=1, nz=1,
                   xmax=0.0, xmin=0.0, ymin=0, ymax=0, zmax=0, zmin=0, p0=[0, 0, 0])
    dofs = beam.addObject("MechanicalObject", name="DOFs", template="Rigid3d")
    beam.addObject("WireBeamInterpolation", name="BeamInterpolation",
                   WireRestShape="@../EdgeTopology/BeamRestShape")
    beam.addObject("AdaptiveBeamForceFieldAndMass", name="BeamForceField",
                   massDensity=0.00000155, interpolation="@BeamInterpolation")
    ctrl = beam.addObject("InterventionalRadiologyController", name="DeployController",
                          template="Rigid3d", instruments="BeamInterpolation",
                          topology="@MeshLines", startingPos=[0, 0, 0, 0, -0.7071068, 0, 0.7071068],
                          xtip=[0], rotationInstrument=[0], step=0.5, speed=0.0,
                          listening=True, controlledInstrument=0)
    beam.addObject("FixedProjectiveConstraint", indices=0, name="FixedConstraint")
    beam.addObject("RestShapeSpringsForceField", name="RestSPForceField",
                   points="@DeployController.indexFirstNode", angularStiffness=1e8, stiffness=1e8)
    deploy = root.addObject(Deploy(ctrl, name="Deploy"))
    return dofs, ctrl, deploy


def main():
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    root = Sofa.Core.Node("root")
    dofs, ctrl, deploy = create_scene(root)
    Sofa.Simulation.init(root)
    x0 = np.array(dofs.position.value, copy=True)

    for _ in range(STEPS):
        Sofa.Simulation.animate(root, root.dt.value)

    x = np.asarray(dofs.position.value)
    xtip = float(ctrl.xtip.value[0])
    moved = float(np.abs(x[:, :3] - x0[:, :3]).max())
    print(f"steps={STEPS} controller_calls={deploy.calls} xtip={xtip:.1f} "
          f"max_displacement={moved:.3f} nodes={len(x)}")

    assert deploy.calls == STEPS, "Python controller callback did not fire every step"
    assert np.isfinite(x).all(), "solution went non-finite"
    assert xtip >= 2.0 * STEPS - 1e-6, "deployment did not advance"
    assert moved > 1.0, "beam did not move -- solver or deployment is inert"
    print("OK")


if __name__ == "__main__":
    main()
